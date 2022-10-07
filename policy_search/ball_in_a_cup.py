"""
Ball in a Cup environment for policy search experiments.

Builds off code written with Johannes Silberbauer and Michael Lutter.
Also builds of experiment design of Pascal Klink.
"""

import multiprocessing
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

import mujoco_py
import numpy as np
from tqdm import tqdm

from utils import NullContext, VideoRenderStream


class BicType(Enum):
    cone = 1
    cylinder = 2


def point_line_dist(x0, x1, x):
    """Compute distance of x to the line passing through x0 and x1
    (https://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html).
    """
    return np.linalg.norm(np.cross(x1 - x0, x0 - x)) / np.linalg.norm(x1 - x0)


def position_along_axis(x0, x1, x):
    """Returns the positions of the orthogonal projection of x on the
    the axis passing through x0 and x1. The return value of 0 means x0
    and a value of 1 indicates x1."""
    return np.dot(x - x0, x1 - x0) / (np.linalg.norm(x1 - x0) ** 2)


def cylinder_contains(x0, x1, r, x):
    """Returns true if the specified point is inside the cylinder
    with axis passing through x0 and x1 and radius r."""
    d = point_line_dist(x0, x1, x)
    t = position_along_axis(x0, x1, x)
    return (d <= r) and (0 <= t <= 1)


def cone_contains(x_tip, x_base, r, x):
    """Returns true if the specified point is inside the cone with
    base radius r, tip position x_tip and position of the center of
    the base circle at x_base."""
    d = point_line_dist(x_tip, x_base, x)
    t = position_along_axis(x_tip, x_base, x)
    return (d <= t * r) and (0 <= t <= 1)


def wam_collision_geom_names():
    return [
        "base_link_convex_geom",
        "shoulder_link_convex_decomposition_p1_geom",
        "shoulder_link_convex_decomposition_p2_geom",
        "shoulder_link_convex_decomposition_p3_geom",
        "shoulder_pitch_link_convex_geom",
        "upper_arm_link_convex_decomposition_p1_geom",
        "upper_arm_link_convex_decomposition_p2_geom",
        "elbow_link_convex_geom",
        "elbow_link_cylinder_geom",
    ]


def ball_in_cup_xml_path(type_: BicType):
    data_dir = Path(__file__).parent.resolve()
    asset_root = data_dir / "robot_descriptions" / "wam"
    if type_ == BicType.cone:
        path_ = asset_root / "ball_in_cup_cone.xml"
    elif type_ == BicType.cylinder:
        path_ = asset_root / "ball_in_cup_cylinder.xml"
    else:
        raise ValueError("Cannot load ball-in-cup xml. " f"Unknown type {type_}")
    assert path_.is_file()
    return path_


class BallInCupSimTrace(object):
    """A data class for keeping track off relevant states during the
    simulation."""

    ball_positions: np.ndarray
    goal_positions: np.ndarray
    goal_final_positions: np.ndarray
    cup_center_top_positions: np.ndarray
    cup_center_bottom_positions: np.ndarray
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    torques: np.ndarray
    error: bool
    constraint_violation: bool
    n_not_executed_steps: int
    n_cool_down_steps: int
    type_: str

    def __init__(self, n_steps: int, type_: BicType):
        """Initialize the trace and all its underlying buffers
        to hold data for the specified number of steps."""
        self.ball_positions = np.zeros((n_steps, 3))
        self.goal_positions = np.zeros((n_steps, 3))
        self.goal_final_positions = np.zeros((n_steps, 3))
        self.cup_center_top_positions = np.zeros((n_steps, 3))
        self.cup_center_bottom_positions = np.zeros((n_steps, 3))
        self.joint_positions = np.zeros((n_steps, 4))
        self.joint_velocities = np.zeros((n_steps, 4))
        self.torques = np.zeros((n_steps, 4))
        self.error = False
        self.constraint_violation = False
        self.n_not_executed_steps = 0
        self.n_cool_down_steps = 0
        self.type_ = type_

    def subset(self, last_index):
        """Return a copy of the trace containing all buffer values up
        until the specified step index."""
        other = BallInCupSimTrace(last_index, self.type_)
        other.ball_positions = self.ball_positions[:last_index]
        other.goal_positions = self.goal_positions[:last_index]
        other.goal_final_positions = self.goal_final_positions[:last_index]
        other.cup_center_top_positions = self.cup_center_top_positions[:last_index]
        other.cup_center_bottom_positions = self.cup_center_bottom_positions[
            :last_index
        ]
        other.joint_positions = self.joint_positions[:last_index]
        other.joint_velocities = self.joint_velocities[:last_index]
        other.torques = self.torques[:last_index]
        other.error = self.error
        other.constraint_violation = self.constraint_violation
        other.n_not_executed_steps = self.ball_positions.shape[0] - last_index
        other.n_cool_down_steps = self.n_cool_down_steps

        return other

    def copy(self):
        numpy_attrs_names = [
            "ball_positions",
            "goal_positions",
            "goal_final_positions",
            "cup_center_top_positions",
            "cup_center_bottom_positions",
            "joint_positions",
            "joint_velocities",
            "torques",
        ]
        immutable_types_attrs_names = [
            "error",
            "constraint_violation",
            "n_not_executed_steps",
            "n_cool_down_steps",
            "type_",
        ]
        result = BallInCupSimTrace(self.ball_positions.shape[0], self.type_)

        # call numpy copy on numpy arrays
        for attr in numpy_attrs_names:
            setattr(result, attr, np.copy(getattr(self, attr)))

        # simply set attributes on immutable types
        for attr in immutable_types_attrs_names:
            setattr(result, attr, getattr(self, attr))

        return result


class BallInCupSim:
    """MuJoCo wrapper that handles running ball-in-cup simulations."""

    def __init__(self, type_=BicType.cylinder):
        # load MuJoCo model
        self.type_ = type_
        self.xml_path = ball_in_cup_xml_path(type_)
        self.model = mujoco_py.load_model_from_path(str(self.xml_path))
        self.n_sub_steps = 4
        self.sim = self._init_sim()

        # time steps (note: mujoco-py does nsubsteps MuJoCo steps at each call to step;
        # this produces an effectively larger step when viewed from mujoco-py)
        self.dt = self.sim.model.opt.timestep
        self.effective_dt = self.dt * self.n_sub_steps

        # store some common handles
        self.ball_id = self.sim.model.body_name2id("ball")
        self.goal_id = self.sim.model.site_name2id("cup_goal")
        self.goal_final_id = self.sim.model.site_name2id("cup_goal_final")
        self.cup_center_top_id = self.sim.model.site_name2id("cup_center_top")
        self.cup_center_bottom_id = self.sim.model.site_name2id("cup_center_bottom")
        self.ball_col_id = self.sim.model.geom_name2id("ball_geom")
        self.robot_col_ids = [
            self.sim.model.geom_name2id(name) for name in wam_collision_geom_names()
        ]

        # PD controller parameters
        self.p_gains = np.array([200.0, 300.0, 100.0, 100.0])
        self.d_gains = np.array([7, 15, 5, 2.5])

    def _init_sim(self):
        return mujoco_py.MjSim(self.model, nsubsteps=self.n_sub_steps)

    def reset(self, q, recreate_sim=False):
        """Reset the simulation to the specified WAM joint states."""
        if recreate_sim:
            self.sim = self._init_sim()
        sim_state = self.sim.get_state()

        sim_state.qpos[:4] = q
        sim_state.qpos[4:] = 0.0  # reset string
        sim_state.qvel[:] = 0.0  # zero all velocities

        self.sim.set_state(sim_state)
        self.sim.forward()

    def get_joint_states(self):
        """Retrieve WAM joint positions and velocities."""
        return (self.sim.data.qpos[:4].copy(), self.sim.data.qvel[:4].copy())

    def set_named_joint_pos(self, joint_name, pos):
        """Modifies the joint positions of the given named joint."""
        index = self.sim.model.get_joint_qpos_addr(joint_name)
        sim_state = self.sim.get_state()

        if isinstance(index, int):
            sim_state.qpos[index] = pos
        else:
            sim_state.qpos[index[0] : index[1]] = pos

        self.sim.set_state(sim_state)

    def get_task_state(self):
        """Get the state of the task defined by the tuple of ball, goal and
        cup positions."""
        return (
            self.sim.data.body_xpos[self.ball_id],
            self.sim.data.site_xpos[self.goal_id],
            self.sim.data.site_xpos[self.goal_final_id],
            self.sim.data.site_xpos[self.cup_center_bottom_id],
            self.sim.data.site_xpos[self.cup_center_top_id],
        )

    def update_trace(self, idx_step, trace):
        """Write current joint states and tracked task states to the specified trace index."""
        (
            trace.ball_positions[idx_step],
            trace.goal_positions[idx_step],
            trace.goal_final_positions[idx_step],
            trace.cup_center_bottom_positions[idx_step],
            trace.cup_center_top_positions[idx_step],
        ) = self.get_task_state()
        (
            trace.joint_positions[idx_step],
            trace.joint_velocities[idx_step],
        ) = self.get_joint_states()

    def render_camera_img(self, frame_size):
        """Render a default camera view of the scene."""
        return np.flipud(
            self.sim.render(
                width=frame_size[0],
                height=frame_size[1],
                camera_name="side",
                depth=False,
            )
        ).copy(order="C")

    def check_ball_robot_collision(self):
        """Test whether there's a collision between the ball and any
        other part of the robot."""
        for idx_contact in range(self.sim.data.ncon):
            contact = self.sim.data.contact[idx_contact]

            collision = (
                contact.geom1 in self.robot_col_ids
                and contact.geom2 == self.ball_col_id
            )
            collision_trans = (
                contact.geom1 == self.ball_col_id
                and contact.geom2 in self.robot_col_ids
            )

            if collision or collision_trans:
                return True
        return False

    def stabilize_current_pos(self):
        """Runs the PD controller for multiple steps. The set-point is
        the current joint state and zero velocities."""
        q0 = self.get_joint_states()[0]
        qd0 = np.zeros_like(q0)

        for idx_step in range(500):
            self.set_setpoint(q0, qd0)
            self.step()

    def execute_trajectory(
        self,
        qs,
        qds,
        stabilize_current_pos=True,
        cool_down_final_pos=True,
        verbose=False,
        video_writer=None,
        overwrite_jnt_states=None,
    ):
        """Execute a trajectory of desired WAM joint positions and
        velocities using PD control."""
        n_steps = qs.shape[0] + (350 if cool_down_final_pos else 0)
        steps_per_video_frame = (
            int((1 / self.effective_dt) / video_writer.fps)
            if video_writer is not None
            else 1
        )

        # if needed stabilize the current position before
        # executing the trajectory
        if stabilize_current_pos:
            self.stabilize_current_pos()

        # output buffers
        trace = BallInCupSimTrace(n_steps, self.type_)

        pbar = tqdm(
            range(n_steps), desc="Simulating", unit="Steps", disable=(not verbose)
        )
        for idx_step in pbar:
            # retrieve system states
            self.update_trace(idx_step, trace)

            # apply controls
            idx_trj = min(qs.shape[0] - 1, idx_step)
            trace.n_cool_down_steps = max(idx_step - idx_trj, 0)
            qd_set = qds[idx_trj] if qds is not None else None
            trace.torques[idx_step] = self.set_setpoint(qs[idx_trj], qd_set)

            # overwrite requested joint states
            if overwrite_jnt_states is not None:
                for joint_name, states in overwrite_jnt_states.items():
                    self.set_named_joint_pos(
                        joint_name, states[min(states.shape[0] - 1, idx_trj)]
                    )

            # advance simulation, in the case of an error return trace
            # (e.g. divergent simulation) recorded so far
            try:
                self.step()
            except mujoco_py.builder.MujocoException as ex:
                trace.error = True
                if verbose:
                    pbar.write(
                        "Simulation at {}s encountered and error ({}). "
                        "Stopping early.".format(idx_step * self.effective_dt, repr(ex))
                    )
                return trace.subset(idx_step)

            # also end the simulation if the ball hits the robot
            if self.check_ball_robot_collision():
                trace.constraint_violation = True
                if verbose:
                    pbar.write(
                        "Simulation at {}s violated constraints (ball hit robot). "
                        "Stopping early.".format(idx_step * self.effective_dt)
                    )
                return trace.subset(idx_step)

            # render a video frame if needed and requested
            if video_writer is not None and (idx_step % steps_per_video_frame == 0):
                video_writer.write(self.render_camera_img(video_writer.frame_size))

        return trace

    def set_setpoint(self, q, qd):
        """Set joint torques using PD control. Note that if you don't
        specify a desired velocity only the P-part of the controller
        is active."""
        q_current, qd_current = self.get_joint_states()
        tau_d = np.zeros_like(q) if qd is None else self.d_gains * (qd - qd_current)
        tau = self.p_gains * (q - q_current) + tau_d
        self.set_torques(tau)
        return tau

    def set_torques(self, tau):
        self.sim.data.qfrc_applied[:4] = tau

    def step(self):
        self.sim.step()

    def replay(
        self,
        trace,
        video_writer,
        true_ball_positions=None,
        show_trace_cup_position=False,
    ):
        """Render a video of the joint and ball positions in the trace. This does not
        render the string position since it's not well-defined from the ball position.

        Note that the input positions should be relative to the MuJoCo world frame.
        """
        n_steps = trace.joint_positions.shape[0]
        steps_per_video_frame = int((1 / self.effective_dt) / video_writer.fps)
        ball_pred_q_index = self.sim.model.get_joint_qpos_addr("ball_pred_jnt")
        ball_true_q_index = self.sim.model.get_joint_qpos_addr("ball_true_jnt")
        cup_pred_q_index = self.sim.model.get_joint_qpos_addr("cup_pred_jnt")
        identity_quaternion = np.array([1.0, 0.0, 0.0, 0.0])

        for idx_step in range(n_steps):
            sim_state = self.sim.get_state()

            jnt_pos = trace.joint_positions[idx_step, ...]
            jnt_pos = (
                jnt_pos.detach().clone()
                if not isinstance(jnt_pos, np.ndarray)
                else jnt_pos.copy()
            )
            sim_state.qpos[:4] = jnt_pos
            sim_state.qpos[4:] = 0.0  # reset string

            ball_pos = trace.ball_positions[idx_step, ...]
            ball_pos = (
                ball_pos.detach().clone()
                if not isinstance(ball_pos, np.ndarray)
                else ball_pos.copy()
            )

            ball_pos_pred = np.concatenate((ball_pos, identity_quaternion))
            sim_state.qpos[ball_pred_q_index[0] : ball_pred_q_index[1]] = ball_pos_pred
            if true_ball_positions is not None:
                ball_pos_true = true_ball_positions[idx_step, ...]
                world_to_ball_true = np.concatenate(
                    (ball_pos_true, identity_quaternion)
                )
                sim_state.qpos[
                    ball_true_q_index[0] : ball_true_q_index[1]
                ] = world_to_ball_true

            if show_trace_cup_position:
                cup_pos_pred = trace.cup_center_bottom_positions[idx_step, ...]
                world_to_cup_pred = np.concatenate((cup_pos_pred, identity_quaternion))
                sim_state.qpos[
                    cup_pred_q_index[0] : cup_pred_q_index[1]
                ] = world_to_cup_pred

            self.sim.set_state(sim_state)
            self.sim.forward()

            # render a video frame if needed and requested
            if video_writer is not None and (idx_step % steps_per_video_frame == 0):
                video_writer.write(self.render_camera_img(video_writer.frame_size))


def evaluate_trajectory(q0, trj, trj_kwargs, sim_init_kwargs):
    if sim_init_kwargs is None:
        sim_init_kwargs = {}

    # init the simulation
    sim = BallInCupSim(**sim_init_kwargs)
    sim.reset(q0)

    # simulate the trajectory and return the recorded trace
    return sim.execute_trajectory(*trj, **trj_kwargs)


class BallInCupParallelTrajectoryEvaluator:
    """A class for handling the evaluation of multiple trajectories using
    pythons multiprocessing tools."""

    def __init__(self, q0):
        self.q0 = q0

    def eval(self, trajectories, trajectory_exe_kwargs, sim_init_kwargs):
        n_tasks = (
            len(trajectories)
            if isinstance(trajectories, list)
            else trajectories.shape[0]
        )
        pool = multiprocessing.Pool(multiprocessing.cpu_count())

        results = []
        for idx_task in range(n_tasks):
            results.append(
                pool.apply_async(
                    func=evaluate_trajectory,
                    args=(
                        self.q0,
                        trajectories[idx_task],
                        trajectory_exe_kwargs,
                        sim_init_kwargs,
                    ),
                )
            )

        results = [r.get() for r in results]
        pool.close()
        return results


def _test_trajectory(dt, t_end):
    ts = np.arange(int(t_end / dt)) * dt

    # construct position trajectories
    max_pos = 1.8
    pos_mod = np.linspace(0.6, max_pos / 2, ts.shape[0])
    freq_mod = np.linspace(0.3, 0.6, ts.shape[0])
    qs = np.zeros((ts.shape[0], 4))
    qs[:, 0] = pos_mod * (np.sin(ts * 2.0 * np.pi * freq_mod))
    qs[:, 3] = 1.57
    return qs, None


def compute_state_reward_dipole_potential(trace: BallInCupSimTrace, **kwargs):
    """Computes reward for ball positions based a potential function
    similar to the magnetic scalar potential of a dipole placed at
    the bottom of the cup."""
    eps, beta = kwargs["dipole_eps"], kwargs["dipole_beta"]
    min_weight = kwargs["min_weight"]

    m = trace.cup_center_top_positions - trace.cup_center_bottom_positions
    m /= np.linalg.norm(m, axis=-1)[..., None]

    rm = trace.ball_positions - trace.cup_center_top_positions
    pot_m = (rm * m).sum(-1) / (np.linalg.norm(rm, axis=-1) ** 2 + eps)

    rl = trace.ball_positions - trace.cup_center_bottom_positions
    pot_l = (rl * m).sum(-1) / (np.linalg.norm(rl, axis=-1) ** 2 + eps)

    return np.exp(
        min_weight * np.max(pot_m * beta) + (1.0 - min_weight) * (pot_l * beta)[-1]
    )


def compute_state_reward_euclidean(trace, **kwargs):
    """Computes rewards for ball positions based on euclidean distance
    to desired positions."""
    min_weight = kwargs["min_weight"]

    dists = np.linalg.norm(trace.goal_positions - trace.ball_positions, axis=-1)
    dists_final = np.linalg.norm(
        trace.goal_final_positions - trace.ball_positions, axis=-1
    )
    min_dist = min_weight * np.min(dists) + (1 - min_weight) * dists_final[-1]
    return np.exp(-2.0 * min_dist)


def compute_state_reward(trace, type_, reward_kwargs):
    """Computes reward for the tasks states i.e. ball positions."""
    reward_fns = {
        "dipole_potential": compute_state_reward_dipole_potential,
        "euclidean": compute_state_reward_euclidean,
    }
    return reward_fns[type_](trace, **reward_kwargs)


def compute_joint_velocity_penalty(trace):
    """Compute the unweighted penalty term for joint velocities."""
    vel = trace.joint_velocities
    if trace.n_not_executed_steps > 0:
        # copy last torque as if it would have been applied until the end of the trajectory
        # INSPECT: This is taken from https://github.com/psclklnk/self-paced-rl - check
        #  whether this is necessary.
        not_executed_action = np.tile(vel[-1], trace.n_not_executed_steps).reshape(
            trace.n_not_executed_steps, vel.shape[-1]
        )
        vel = np.concatenate((vel, not_executed_action))

    return np.mean(np.sum(np.square(vel), axis=1), axis=0)


def compute_joint_position_penalty(trace):
    """Compute the unweighted penalty term for joint positions."""
    pos_offset = trace.joint_positions - trace.joint_positions[0]
    return np.mean((pos_offset * pos_offset).sum(-1))


def compute_ball_velocity_penalty(trace):
    """Compute the unweighted penalty term for ball velocities."""
    dt = 1 / 500.0  # TODO: Don't hard-code this.
    velocities = np.diff(trace.ball_positions, axis=0) / dt
    return np.mean((velocities * velocities).sum(-1))


def compute_success(trace, cup_inner_radius):
    """Compute whether the last ball position is inside the shape formed
    by the inner sides of the cup."""
    if trace.type_ == "cylinder":
        return cylinder_contains(
            trace.cup_center_bottom_positions[-1],
            trace.cup_center_top_positions[-1],
            cup_inner_radius,
            trace.ball_positions[-1],
        )
    elif trace.type_ == "cone":
        return cone_contains(
            trace.cup_center_bottom_positions[-1],
            trace.cup_center_top_positions[-1],
            cup_inner_radius,
            trace.ball_positions[-1],
        )
    else:
        # if the cup shape is unknown threshold the distance to the bottom of the cup
        ball_to_goal = np.linalg.norm(
            trace.cup_center_bottom_positions[-1] - trace.ball_positions[-1]
        )
        return ball_to_goal < cup_inner_radius


@dataclass
class BallInCupRewardParams:
    """Parameters for the reward / success function of ball-in-cup."""

    state_reward_type: str
    joint_velocity_penalty_factor: float
    joint_position_penalty_factor: float
    ball_velocity_penalty_factor: float
    cup_inner_radius: float
    reward_dipole_eps: float
    reward_dipole_beta: float
    reward_min_weight: float

    def state_reward_kwargs(self):
        return {
            "dipole_eps": self.reward_dipole_eps,
            "dipole_beta": self.reward_dipole_beta,
            "min_weight": self.reward_min_weight,
        }

    def as_dict(self):
        return asdict(self)


def compute_reward(trace, params):
    """Compute reward and success flag for a single trace."""
    state_reward = compute_state_reward(
        trace, params.state_reward_type, params.state_reward_kwargs()
    )
    joint_pos_cost = (
        params.joint_position_penalty_factor * compute_joint_position_penalty(trace)
    )
    joint_vel_cost = (
        params.joint_velocity_penalty_factor * compute_joint_velocity_penalty(trace)
    )
    ball_vel_cost = params.ball_velocity_penalty_factor * compute_ball_velocity_penalty(
        trace
    )

    reward = state_reward - joint_pos_cost - joint_vel_cost - ball_vel_cost
    success = compute_success(trace, params.cup_inner_radius)
    return reward, success


def compute_rewards(traces, params):
    """Compute rewards for a list of traces."""
    rewards, success_flags = [], []
    for trace in traces:
        reward, success = compute_reward(trace, params)
        rewards.append(reward)
        success_flags.append(success)

    return rewards, success_flags


if __name__ == "__main__":
    t_end = 20
    dt = BallInCupSim(BicType.cylinder).effective_dt
    q0 = np.array([0.0, 0.0, 0.0, 1.5707])

    # run multiple simulations in parallel
    evaluator = BallInCupParallelTrajectoryEvaluator(q0)
    trajectories = [_test_trajectory(dt, t_end) for _ in range(12)]
    traces = evaluator.eval(
        trajectories=trajectories,
        trajectory_exe_kwargs=dict(
            stabilize_current_pos=True, verbose=False, video_writer=None
        ),
        sim_init_kwargs=dict(type_=BicType.cylinder),
    )

    # run a single simulation (optionally rendering a video)
    # render_video = False
    render_video = True
    video_render_ctx = (
        VideoRenderStream("ball-in-cup_TEMP.mp4", Path(__file__).parent.resolve())
        if render_video
        else NullContext()
    )
    with video_render_ctx as video_stream:
        exe_params = dict(
            stabilize_current_pos=True, verbose=False, video_writer=video_stream
        )
        trace = evaluate_trajectory(
            q0=q0,
            trj=_test_trajectory(dt, t_end),
            trj_kwargs=exe_params,
            sim_init_kwargs=dict(type_=BicType.cylinder),
        )

    print("Done.")
