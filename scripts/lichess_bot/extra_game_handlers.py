"""Functions for the user to implement when the config file is not adequate to express bot requirements."""
import logging

from lib import model
from lib.lichess_types import OPTIONS_TYPE

logger = logging.getLogger(__name__)

# PieBot's Contempt is the score, in centipawns, the engine gives a draw for
# itself. Positive values make it steer towards draws (useful against stronger
# opponents: 40 measurably helped against Stockfish 19); negative values make
# it avoid draws (useful against weaker opponents).
CONTEMPT_PER_RATING_POINT = 1 / 5   # +200 rating difference -> +40
CONTEMPT_MIN = -25                  # how hard to play for a win against weaker players
CONTEMPT_MAX = 50                   # how hard to hold a draw against stronger players


def contempt_for(my_rating: int | None, opponent_rating: int | None, opponent_provisional: bool = False) -> int:
    """
    Contempt for a game, from the two players' lichess ratings in the game's rating pool.

    The rating difference is mapped linearly and clamped. Unknown or provisional
    opponent ratings say little about strength, so they get no contempt.
    """
    if my_rating is None or opponent_rating is None or opponent_provisional:
        return 0
    contempt = round((opponent_rating - my_rating) * CONTEMPT_PER_RATING_POINT)
    return max(CONTEMPT_MIN, min(CONTEMPT_MAX, contempt))


def game_specific_options(game: model.Game) -> OPTIONS_TYPE:
    """
    Return a dictionary of engine options based on game aspects.

    Sets PieBot's Contempt from the opponent's lichess rating relative to the bot's.
    """
    contempt = contempt_for(game.me.rating, game.opponent.rating, bool(game.opponent.provisional))
    logger.info(f"Contempt {contempt} for game {game.id}: bot {game.me.rating}, "
                f"{game.opponent.name} {game.opponent.rating}{'?' if game.opponent.provisional else ''}")
    return {"Contempt": contempt}


def is_supported_extra(challenge: model.Challenge) -> bool:  # noqa: ARG001
    """
    Determine whether to accept a challenge.

    By default, True is always returned so that there are no extra restrictions beyond those in the config file.
    """
    return True
