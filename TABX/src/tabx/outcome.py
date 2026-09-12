"""Terminal outcome rules for a two-team TABX battle."""


def did_ally_win(team_dones, truncation):
    """Return whether team 0 won the episode.

    Team 0 wins only by eliminating team 1 before the episode limit. Every
    other terminal outcome is a team-0 loss; there is no draw result.
    """

    return team_dones[1] & (~truncation.ravel()[0])
