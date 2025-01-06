import os
import pandas as pd
from pandas.api.types import is_object_dtype


def convert_to_minutes(time_str):
    """
    Convert time string in format 'MM:SS' to minutes (float)

    Args:
        time_str (str): Time string in format 'MM:SS'

    Returns:
        float: Total minutes including fractional minutes from seconds
    """
    try:
        minutes, seconds = map(int, time_str.split(":"))
        total_minutes = minutes + (seconds / 60)
        return total_minutes
    except:
        return None


def height_to_inches(height_series):
    """
    Convert height series in format 'feet-inches' to total inches

    Args:
        height_series: pandas Series of strings in format 'feet-inches' (e.g. '6-2')

    Returns:
        pandas Series: total inches as integers
    """
    # Split the string series into two columns
    feet, inches = height_series.str.split("-", expand=True).astype(float)

    # Convert to total inches
    return int(feet * 12 + inches)


def filter_by_game_play_ids(df, id_pairs):
    """
    Filter DataFrame by tuples of (game_id, play_id, nfl_id) using a join-based approach
    while maintaining order of id_pairs

    Args:
        df: pandas DataFrame with gameId, playId, and nflId columns
        id_pairs: list of tuples [(game_id, play_id, nfl_id), ...]
    """
    # Convert tuple elements to integers first
    id_pairs = [
        (int(game_id), int(play_id), int(nfl_id))
        for game_id, play_id, nfl_id in id_pairs
    ]

    # Create DataFrame from integer tuples
    pairs_df = pd.DataFrame(
        id_pairs, columns=["gameId", "playId", "nflId"]
    ).reset_index()

    # Merge while keeping the position index
    result = (
        df.merge(pairs_df, on=["gameId", "playId", "nflId"], how="inner")
        .sort_values("index")
        .drop("index", axis=1)
    )

    return result


BASE_PATH = "."


def load_and_filter_data(weeks, offense_only=True, base_path=BASE_PATH):

    tracking = pd.read_csv(
        os.path.join(base_path, f"nfl-big-data-bowl-2025/tracking_week_{weeks[0]}.csv")
    )
    for week in weeks[1:]:
        tracking = pd.concat(
            [
                tracking,
                pd.read_csv(
                    os.path.join(
                        base_path, f"nfl-big-data-bowl-2025/tracking_week_{week}.csv"
                    )
                ),
            ]
        )
        tracking = tracking[tracking.event == "ball_snap"]
    games = pd.read_csv(os.path.join(base_path, "nfl-big-data-bowl-2025/games.csv"))
    plays = pd.read_csv(os.path.join(base_path, "nfl-big-data-bowl-2025/plays.csv"))

    player_play_data = pd.read_csv(
        os.path.join(base_path, "nfl-big-data-bowl-2025/player_play.csv")
    )
    print("loaded")
    offensive_positions = ("QB", "RB", "FB", "WR", "TE", "C", "G", "T", "OL")
    players = pd.read_csv(os.path.join(base_path, "nfl-big-data-bowl-2025/players.csv"))
    if offense_only:
        players = players[players.position.isin(offensive_positions)]
    player_play_data = player_play_data[
        player_play_data.gameId.isin(tracking.gameId.unique())
    ]
    tracking = pd.merge(tracking, player_play_data, on=["nflId", "playId", "gameId"])
    tracking["nflId"] = tracking["nflId"].apply(lambda x: int(x))
    print("merged player play")
    tracking = pd.merge(tracking, plays, on=["playId", "gameId"])
    print("merged plays")
    tracking = pd.merge(tracking, players, on="nflId")
    tracking = pd.merge(tracking, games, on=["gameId"])
    tracking["height_inches"] = tracking.height.pipe(height_to_inches)

    unique_ids = tracking["nflId"].unique()
    id_mapping = {id_: idx for idx, id_ in enumerate(unique_ids)}
    # Map the nflId column
    tracking["nflId"] = tracking["nflId"].map(id_mapping)

    if is_object_dtype(tracking.gameClock.dtype):
        tracking.gameClock = tracking.gameClock.apply(convert_to_minutes)

    # Handle missing values based on data type and missingness percentage
    numeric_columns = tracking.select_dtypes(include=["int64", "float64"]).columns
    categorical_columns = tracking.select_dtypes(include=["object"]).columns

    # Handle numeric columns
    for column in numeric_columns:
        missing_percentage = (tracking[column].isna().sum() / len(tracking)) * 100

        if missing_percentage > 0:  # Only process columns with missing values
            if missing_percentage <= 10:
                # Fill missing values with column mean
                column_mean = tracking[column].mean()
                tracking[column] = tracking[column].fillna(column_mean)
                print(
                    f"Filled {missing_percentage:.2f}% missing values in {column} with mean"
                )
            else:
                # Drop columns with more than 10% missing values
                tracking = tracking.drop(columns=[column])
                print(
                    f"Dropped {column} due to {missing_percentage:.2f}% missing values"
                )

    # Handle categorical columns
    for column in categorical_columns:
        if column == "routeRan":
            continue
        missing_percentage = (tracking[column].isna().sum() / len(tracking)) * 100

        if missing_percentage > 0:  # Only process columns with missing values
            if missing_percentage <= 10:
                # Fill missing values with most frequent value
                mode_value = tracking[column].mode()[0]
                tracking[column] = tracking[column].fillna(mode_value)
                print(
                    f"Filled {missing_percentage:.2f}% missing values in {column} with mode"
                )
            else:
                # Keep missing values as N/A for categorical columns with >10% missing
                print(
                    f"Keeping {missing_percentage:.2f}% missing values in {column} as N/A"
                )

    print("Missing value handling completed")
    return tracking


def filter_passing_plays_only(tracking):
    passing_plays = (
        tracking[tracking.isDropback == 1]
        .reset_index()[["gameId", "playId"]]
        .drop_duplicates()
    )
    print(passing_plays.shape)
    print(tracking.shape)
    tracking_passing_only = pd.merge(
        tracking, passing_plays, on=["gameId", "playId"], how="inner"
    )
    print(tracking_passing_only.shape)
    return tracking_passing_only


def get_game_scores(row):
    """
    Determine offensive and defensive scores based on possession and team information.

    Parameters:
    row: pandas Series containing game state information

    Returns:
    tuple: (offense_score, defense_score, possession_team)
    """
    possession_is_home = row["possessionTeam"] == row["homeTeamAbbr"]

    if possession_is_home:
        offense_score = row["preSnapHomeScore"]
        defense_score = row["preSnapVisitorScore"]
    else:
        offense_score = row["preSnapVisitorScore"]
        defense_score = row["preSnapHomeScore"]

    return offense_score, defense_score, row["possessionTeam"]


def process_game_scores(df):
    """
    Process entire dataframe to add offensive and defensive scores columns.

    Parameters:
    df: pandas DataFrame containing game state information

    Returns:
    pandas DataFrame: Original dataframe with two new columns added
    """
    # Create copy to avoid modifying original dataframe
    result_df = df.copy()

    # Apply get_game_scores to each row and create new columns
    result_df[["offenseScore", "defenseScore", "offenseTeam"]] = pd.DataFrame(
        result_df.apply(get_game_scores, axis=1).tolist(), index=result_df.index
    )

    return result_df


def create_game_play_pairs(df):
    return list(df.groupby(["gameId", "playId"]).groups.keys())
