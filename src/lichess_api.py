"""
lichess_api.py

Fetches a player's recent games directly from the public Lichess API
(https://lichess.org/api/games/user/{username}), as an alternative to
asking them to export and upload a PGN file by hand.

No authentication needed -- this only reads a user's own PUBLIC game
history, same as browsing their profile on lichess.org. The API enforces
"one request at a time" per client and will return 429 if that's
violated; this module makes a single request and surfaces that error
clearly rather than retrying aggressively against someone else's server.
"""

import requests

API_URL_TEMPLATE = "https://lichess.org/api/games/user/{username}"
USER_AGENT = "chess-rating-predictor (https://github.com/EloiGomez/chess-rating-predictor)"
REQUEST_TIMEOUT = 30  # seconds; this endpoint streams, a slow connection can legitimately take a while


class LichessApiError(Exception):
    """Raised for any non-2xx response, with a message meant to be shown
    directly to the app's user."""


def fetch_games_pgn(username: str, perf_type: str, max_games: int = 20) -> str:
    """Downloads up to `max_games` of a player's own rated games in the
    given perf type (bullet/blitz/rapid/classical), as raw PGN text
    (including clock comments, needed for the time-per-move feature).

    Returns an empty string if the player has no games in that perf type
    (not an error -- the caller should treat that as "not enough games").
    """
    url = API_URL_TEMPLATE.format(username=username)
    params = {
        "max": max_games,
        "perfType": perf_type,
        "rated": "true",
        "clocks": "true",
        "opening": "true",
    }
    headers = {
        "Accept": "application/x-chess-pgn",
        "User-Agent": USER_AGENT,
    }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as error:
        raise LichessApiError(f"Couldn't reach the Lichess API: {error}") from error

    if response.status_code == 404:
        raise LichessApiError(f"Lichess user '{username}' not found.")
    if response.status_code == 429:
        raise LichessApiError(
            "Lichess is rate-limiting this app right now (\"one request at a time\"). "
            "Wait a few seconds and try again."
        )
    if not response.ok:
        raise LichessApiError(f"Lichess API returned {response.status_code}: {response.text[:200]}")

    return response.text
