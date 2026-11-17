"""Parameters of architecture A, each with its physical rationale.

None of these values were fitted to, or chosen by looking at, any ground truth for this match
(including the public five-minute window). They are physical or football priors and must only
be revised using permitted external validation data. Record every change in docs/HANDOFF.md.
"""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Config:
    # --- control evidence -------------------------------------------------------------------
    # A player controlling the ball is normally within about 1-1.5 m of it.
    control_radius_m: float = 1.5
    control_softness_m: float = 0.5
    # A ball faster than ~8 m/s is a pass/shot/clearance in flight, not a ball being carried
    # (sprinting players top out around 9-10 m/s and rarely carry the ball that fast).
    carry_speed_mps: float = 8.0
    carry_speed_softness_mps: float = 1.5
    # Strength with which net control evidence shifts the H-vs-A probability.
    control_gain: float = 3.0
    # Ball speed is noisy frame to frame; smooth it over +/- this many frames (0.2 s).
    speed_half_window: int = 5

    # --- dead-ball evidence (log-odds contributions) ----------------------------------------
    dead_base_logit: float = -1.0
    # Whole ball over a line. Ball radius is 0.11 m; extra margin absorbs tracking noise.
    out_of_bounds_margin_m: float = 0.3
    dead_out_of_bounds: float = 3.0
    # During stoppages players walk; in live play the median outfield speed is rarely this low
    # for long. Soft, because calm build-up can also be slow.
    slow_players_mps: float = 0.8
    slow_players_softness_mps: float = 0.25
    dead_slow_players: float = 1.5
    # A stationary ball (set piece placement) when it is seen.
    still_ball_mps: float = 0.5
    dead_still_ball: float = 0.75
    # Smart Tagging set-piece intervals surround a restart (they open a few seconds early).
    dead_set_piece_tag: float = 1.0
    # Almost no tracked players means the camera/tracker is not on live play.
    min_players_for_play: int = 8
    dead_few_players: float = 2.0
    # Clear control by a moving player argues against a stoppage.
    alive_control: float = 1.0

    # --- Smart Tagging team priors (weak: phases are not possessions) -----------------------
    tag_phase_gain: float = 0.3
    tag_set_piece_gain: float = 0.6

    # --- temporal model ---------------------------------------------------------------------
    # Expected lengths are football priors, not statistics of this match's ground truth.
    mean_team_state_s: float = 12.0   # time in one team state before it ends
    turnover_fraction: float = 0.5    # share of team-state ends that are turnovers vs stoppages
    mean_dead_s: float = 20.0         # typical stoppage length
    # Consecutive frames at 25 fps are strongly correlated; temper per-frame evidence so that
    # ~1 s of clear, sustained opposing control is needed to switch teams, not one touch.
    evidence_temperature: float = 0.1

    def as_dict(self):
        return asdict(self)


DEFAULT = Config()
