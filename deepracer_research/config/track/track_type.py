from enum import Enum
from typing import List


class TrackType(str, Enum):
    """Enumeration of available AWS DeepRacer tracks for training and evaluation"""

    REINVENT_2024_CHAMP = "2024_reinvent_champ"

    REINVENT_2022_CHAMP = "2022_reinvent_champ"
    OCTOBER_2022_PRO = "2022_october_pro"
    OCTOBER_2022_OPEN = "2022_october_open"
    SEPTEMBER_2022_PRO = "2022_september_pro"
    SEPTEMBER_2022_OPEN = "2022_september_open"
    AUGUST_2022_PRO = "2022_august_pro"
    AUGUST_2022_OPEN = "2022_august_open"
    JULY_2022_PRO = "2022_july_pro"
    JULY_2022_OPEN = "2022_july_open"
    JUNE_2022_PRO = "2022_june_pro"
    JUNE_2022_OPEN = "2022_june_open"
    MAY_2022_OPEN = "2022_may_open"
    MAY_2022_PRO = "2022_may_pro"
    APRIL_2022_PRO = "2022_april_pro"
    APRIL_2022_OPEN = "2022_april_open"
    MARCH_2022_OPEN = "2022_march_open"
    MARCH_2022_PRO = "2022_march_pro"
    SUMMIT_2022_SPEEDWAY = "2022_summit_speedway"

    CAECER_LOOP = "caecer_loop"
    CAECER_GP = "caecer_gp"
    RED_STAR_OPEN = "red_star_open"
    RED_STAR_PRO = "red_star_pro"
    MORGAN_PRO = "morgan_pro"
    MORGAN_OPEN = "morgan_open"
    ARCTIC_PRO = "arctic_pro"
    ARCTIC_OPEN = "arctic_open"
    DUBAI_PRO = "dubai_pro"
    DUBAI_OPEN = "dubai_open"
    HAMPTION_OPEN = "hamption_open"
    JYLLANDSRINGEN_PRO = "jyllandsringen_pro"
    JYLLANDSRINGEN_OPEN = "jyllandsringen_open"
    HAMPTION_PRO = "hampton_pro"
    THUNDER_HILL_PRO = "thunder_hill_pro"
    THUNDER_HILL_OPEN = "thunder_hill_open"
    PENBAY_OPEN = "penbay_open"
    PENBAY_PRO = "penbay_pro"

    MONACO_BUILDING = "Monaco_building"
    SINGAPORE_BUILDING = "Singapore_building"

    AUSTIN = "Austin"
    SINGAPORE = "Singapore"
    MONACO = "Monaco"
    ARAGON = "Aragon"
    BELILLE = "Belille"
    ALBERT = "Albert"

    JULY_2020 = "July_2020"
    FS_JUNE_2020 = "FS_June2020"
    SPAIN_TRACK = "Spain_track"

    REINVENT_2019_TRACK = "reInvent2019_track"
    REINVENT_BASE = "reinvent_base"
    AMERICAS_GENERATED = "AmericasGeneratedInclStart"
    LGS_WIDE = "LGSWide"
    VEGAS_TRACK = "Vegas_track"

    CANADA_TRAINING = "Canada_Training"
    MEXICO_TRACK = "Mexico_track"
    CHINA_TRACK = "China_track"
    NEW_YORK_TRACK = "New_York_Track"
    TOKYO_TRAINING_TRACK = "Tokyo_Training_track"
    VIRTUAL_MAY19_TRAIN = "Virtual_May19_Train_track"

    BOWTIE_TRACK = "Bowtie_track"
    OVAL_TRACK = "Oval_track"
    REINVENT_2019_WIDE = "reInvent2019_wide"

    @classmethod
    def get_scenario_tracks(cls, scenario: str) -> "TrackType":
        """Get recommended track for experimental scenario.

        Parameters
        ----------
        scenario : str
            The experimental scenario name

        Returns
        -------
        TrackType
            Recommended track for the scenario
        """
        scenario_mapping = {
            "centerline_following": cls.REINVENT_BASE,
            "static_object_avoidance": cls.OVAL_TRACK,
            "dynamic_object_avoidance": cls.BOWTIE_TRACK,
            "multi_agent": cls.HEAD_TO_HEAD_TRACK,
            "speed_optimization": cls.SPEED_TRACK,
            "time_trial": cls.TIME_TRIAL_TRACK,
            "head_to_head": cls.HEAD_TO_HEAD_TRACK,
        }
        return scenario_mapping.get(scenario, cls.REINVENT_BASE)

    @classmethod
    def get_championship_tracks(cls) -> list["TrackType"]:
        """Get list of championship tracks.

        Returns
        -------
        list[TrackType]
            List of championship track types
        """
        return [cls.REINVENT_2024_CHAMP, cls.REINVENT_2022_CHAMP, cls.REINVENT_BASE, cls.REINVENT_2019_TRACK]

    @classmethod
    def get_professional_tracks(cls) -> list["TrackType"]:
        """Get list of professional tracks.

        Returns
        -------
        list[TrackType]
            List of professional track types
        """
        return [
            cls.JYLLANDSRINGEN_PRO,
            cls.THUNDER_HILL_PRO,
            cls.RED_STAR_PRO,
            cls.PENBAY_PRO,
            cls.ARCTIC_PRO,
            cls.DUBAI_PRO,
            cls.HAMPTION_PRO,
            cls.MORGAN_PRO,
            cls.OCTOBER_2022_PRO,
            cls.SEPTEMBER_2022_PRO,
            cls.AUGUST_2022_PRO,
            cls.JULY_2022_PRO,
            cls.JUNE_2022_PRO,
            cls.MAY_2022_PRO,
            cls.APRIL_2022_PRO,
            cls.MARCH_2022_PRO,
        ]


def get_available_tracks() -> List[str]:
    """Get list of available track ARNs for AWS DeepRacer.

    Returns
    -------
    List[str]
        List of available track ARNs
    """
    return [
        "arn:aws:deepracer:us-east-1::track/2024_reinvent_champ",
        "arn:aws:deepracer:us-east-1::track/2022_reinvent_champ",
        "arn:aws:deepracer:us-east-1::track/2022_october_pro",
        "arn:aws:deepracer:us-east-1::track/2022_october_open",
        "arn:aws:deepracer:us-east-1::track/2022_september_pro",
        "arn:aws:deepracer:us-east-1::track/2022_september_open",
        "arn:aws:deepracer:us-east-1::track/2022_august_pro",
        "arn:aws:deepracer:us-east-1::track/2022_august_open",
        "arn:aws:deepracer:us-east-1::track/2022_july_pro",
        "arn:aws:deepracer:us-east-1::track/2022_july_open",
        "arn:aws:deepracer:us-east-1::track/2022_june_pro",
        "arn:aws:deepracer:us-east-1::track/2022_june_open",
        "arn:aws:deepracer:us-east-1::track/2022_may_open",
        "arn:aws:deepracer:us-east-1::track/2022_may_pro",
        "arn:aws:deepracer:us-east-1::track/2022_april_pro",
        "arn:aws:deepracer:us-east-1::track/2022_april_open",
        "arn:aws:deepracer:us-east-1::track/2022_march_open",
        "arn:aws:deepracer:us-east-1::track/2022_march_pro",
        "arn:aws:deepracer:us-east-1::track/2022_summit_speedway",
        "arn:aws:deepracer:us-east-1::track/caecer_loop",
        "arn:aws:deepracer:us-east-1::track/caecer_gp",
        "arn:aws:deepracer:us-east-1::track/red_star_open",
        "arn:aws:deepracer:us-east-1::track/red_star_pro",
        "arn:aws:deepracer:us-east-1::track/morgan_pro",
        "arn:aws:deepracer:us-east-1::track/morgan_open",
        "arn:aws:deepracer:us-east-1::track/arctic_pro",
        "arn:aws:deepracer:us-east-1::track/arctic_open",
        "arn:aws:deepracer:us-east-1::track/dubai_pro",
        "arn:aws:deepracer:us-east-1::track/dubai_open",
        "arn:aws:deepracer:us-east-1::track/hamption_open",
        "arn:aws:deepracer:us-east-1::track/jyllandsringen_pro",
        "arn:aws:deepracer:us-east-1::track/jyllandsringen_open",
        "arn:aws:deepracer:us-east-1::track/hamption_pro",
        "arn:aws:deepracer:us-east-1::track/thunder_hill_pro",
        "arn:aws:deepracer:us-east-1::track/thunder_hill_open",
        "arn:aws:deepracer:us-east-1::track/penbay_open",
        "arn:aws:deepracer:us-east-1::track/penbay_pro",
        "arn:aws:deepracer:us-east-1::track/Monaco_building",
        "arn:aws:deepracer:us-east-1::track/Singapore_building",
        "arn:aws:deepracer:us-east-1::track/Austin",
        "arn:aws:deepracer:us-east-1::track/Singapore",
        "arn:aws:deepracer:us-east-1::track/Monaco",
        "arn:aws:deepracer:us-east-1::track/Aragon",
        "arn:aws:deepracer:us-east-1::track/Belille",
        "arn:aws:deepracer:us-east-1::track/Albert",
        "arn:aws:deepracer:us-east-1::track/July_2020",
        "arn:aws:deepracer:us-east-1::track/FS_June2020",
        "arn:aws:deepracer:us-east-1::track/Spain_track",
        "arn:aws:deepracer:us-east-1::track/reInvent2019_track",
        "arn:aws:deepracer:us-east-1::track/reinvent_base",
        "arn:aws:deepracer:us-east-1::track/AmericasGeneratedInclStart",
        "arn:aws:deepracer:us-east-1::track/LGSWide",
        "arn:aws:deepracer:us-east-1::track/Vegas_track",
        "arn:aws:deepracer:us-east-1::track/Canada_Training",
        "arn:aws:deepracer:us-east-1::track/Mexico_track",
        "arn:aws:deepracer:us-east-1::track/China_track",
        "arn:aws:deepracer:us-east-1::track/New_York_Track",
        "arn:aws:deepracer:us-east-1::track/Tokyo_Training_track",
        "arn:aws:deepracer:us-east-1::track/Virtual_May19_Train_track",
        "arn:aws:deepracer:us-east-1::track/Bowtie_track",
        "arn:aws:deepracer:us-east-1::track/Oval_track",
        "arn:aws:deepracer:us-east-1::track/reInvent2019_wide",
    ]


def get_track_arn_by_name(track_name: str, region: str = "us-east-1") -> str:
    """Get track ARN by track name.

    Parameters
    ----------
    track_name : str
        Name of the track
    region : str, optional
        AWS region, by default "us-east-1"

    Returns
    -------
    str
        Track ARN

    Raises
    ------
    ValueError
        If track name is not found
    """
    track_mapping = {
        "2024_reinvent_champ": f"arn:aws:deepracer:{region}::track/2024_reinvent_champ",
        "forever_raceway": f"arn:aws:deepracer:{region}::track/2024_reinvent_champ",
        "2022_reinvent_champ": f"arn:aws:deepracer:{region}::track/2022_reinvent_champ",
        "2022_october_pro": f"arn:aws:deepracer:{region}::track/2022_october_pro",
        "2022_october_open": f"arn:aws:deepracer:{region}::track/2022_october_open",
        "2022_september_pro": f"arn:aws:deepracer:{region}::track/2022_september_pro",
        "2022_september_open": f"arn:aws:deepracer:{region}::track/2022_september_open",
        "2022_august_pro": f"arn:aws:deepracer:{region}::track/2022_august_pro",
        "2022_august_open": f"arn:aws:deepracer:{region}::track/2022_august_open",
        "2022_july_pro": f"arn:aws:deepracer:{region}::track/2022_july_pro",
        "2022_july_open": f"arn:aws:deepracer:{region}::track/2022_july_open",
        "2022_june_pro": f"arn:aws:deepracer:{region}::track/2022_june_pro",
        "2022_june_open": f"arn:aws:deepracer:{region}::track/2022_june_open",
        "2022_may_open": f"arn:aws:deepracer:{region}::track/2022_may_open",
        "2022_may_pro": f"arn:aws:deepracer:{region}::track/2022_may_pro",
        "2022_april_pro": f"arn:aws:deepracer:{region}::track/2022_april_pro",
        "2022_april_open": f"arn:aws:deepracer:{region}::track/2022_april_open",
        "2022_march_open": f"arn:aws:deepracer:{region}::track/2022_march_open",
        "2022_march_pro": f"arn:aws:deepracer:{region}::track/2022_march_pro",
        "2022_summit_speedway": f"arn:aws:deepracer:{region}::track/2022_summit_speedway",
        "rl_speedway": f"arn:aws:deepracer:{region}::track/2022_summit_speedway",
        "caecer_loop": f"arn:aws:deepracer:{region}::track/caecer_loop",
        "vivalas_loop": f"arn:aws:deepracer:{region}::track/caecer_loop",
        "caecer_gp": f"arn:aws:deepracer:{region}::track/caecer_gp",
        "vivalas_speedway": f"arn:aws:deepracer:{region}::track/caecer_gp",
        "red_star_open": f"arn:aws:deepracer:{region}::track/red_star_open",
        "red_star_pro": f"arn:aws:deepracer:{region}::track/red_star_pro",
        "morgan_pro": f"arn:aws:deepracer:{region}::track/morgan_pro",
        "morgan_open": f"arn:aws:deepracer:{region}::track/morgan_open",
        "arctic_pro": f"arn:aws:deepracer:{region}::track/arctic_pro",
        "arctic_open": f"arn:aws:deepracer:{region}::track/arctic_open",
        "dubai_pro": f"arn:aws:deepracer:{region}::track/dubai_pro",
        "dubai_open": f"arn:aws:deepracer:{region}::track/dubai_open",
        "hamption_open": f"arn:aws:deepracer:{region}::track/hamption_open",
        "jyllandsringen_pro": f"arn:aws:deepracer:{region}::track/jyllandsringen_pro",
        "jyllandsringen_open": f"arn:aws:deepracer:{region}::track/jyllandsringen_open",
        "hamption_pro": f"arn:aws:deepracer:{region}::track/hamption_pro",
        "thunder_hill_pro": f"arn:aws:deepracer:{region}::track/thunder_hill_pro",
        "thunder_hill_open": f"arn:aws:deepracer:{region}::track/thunder_hill_open",
        "penbay_open": f"arn:aws:deepracer:{region}::track/penbay_open",
        "penbay_pro": f"arn:aws:deepracer:{region}::track/penbay_pro",
        "monaco_building": f"arn:aws:deepracer:{region}::track/Monaco_building",
        "singapore_building": f"arn:aws:deepracer:{region}::track/Singapore_building",
        "austin": f"arn:aws:deepracer:{region}::track/Austin",
        "singapore": f"arn:aws:deepracer:{region}::track/Singapore",
        "monaco": f"arn:aws:deepracer:{region}::track/Monaco",
        "aragon": f"arn:aws:deepracer:{region}::track/Aragon",
        "belille": f"arn:aws:deepracer:{region}::track/Belille",
        "albert": f"arn:aws:deepracer:{region}::track/Albert",
        "reinvent2019_track": f"arn:aws:deepracer:{region}::track/reInvent2019_track",
        "smile_speedway": f"arn:aws:deepracer:{region}::track/reInvent2019_track",
        "reinvent_base": f"arn:aws:deepracer:{region}::track/reinvent_base",
        "reinvent2018": f"arn:aws:deepracer:{region}::track/reinvent_base",
        "bowtie_track": f"arn:aws:deepracer:{region}::track/Bowtie_track",
        "bowtie": f"arn:aws:deepracer:{region}::track/Bowtie_track",
        "oval_track": f"arn:aws:deepracer:{region}::track/Oval_track",
        "oval": f"arn:aws:deepracer:{region}::track/Oval_track",
        "reinvent2019_wide": f"arn:aws:deepracer:{region}::track/reInvent2019_wide",
        "a_to_z_speedway": f"arn:aws:deepracer:{region}::track/reInvent2019_wide",
        "canada_training": f"arn:aws:deepracer:{region}::track/Canada_Training",
        "toronto_turnpike_training": f"arn:aws:deepracer:{region}::track/Canada_Training",
        "mexico_track": f"arn:aws:deepracer:{region}::track/Mexico_track",
        "cumulo_carrera_training": f"arn:aws:deepracer:{region}::track/Mexico_track",
        "china_track": f"arn:aws:deepracer:{region}::track/China_track",
        "shanghai_sudu_training": f"arn:aws:deepracer:{region}::track/China_track",
        "new_york_track": f"arn:aws:deepracer:{region}::track/New_York_Track",
        "empire_city_training": f"arn:aws:deepracer:{region}::track/New_York_Track",
        "tokyo_training_track": f"arn:aws:deepracer:{region}::track/Tokyo_Training_track",
        "kumo_torakku_training": f"arn:aws:deepracer:{region}::track/Tokyo_Training_track",
        "virtual_may19_train_track": f"arn:aws:deepracer:{region}::track/Virtual_May19_Train_track",
        "london_loop_training": f"arn:aws:deepracer:{region}::track/Virtual_May19_Train_track",
        "time_trial": f"arn:aws:deepracer:{region}::track/reInvent2019_track",
        "head_to_head": f"arn:aws:deepracer:{region}::track/reInvent2019_track",
        "speed": f"arn:aws:deepracer:{region}::track/Belille",
    }

    track_arn = track_mapping.get(track_name.lower())
    if not track_arn:
        available_tracks = list(track_mapping.keys())
        raise ValueError(f"Track '{track_name}' not found. Available tracks: {available_tracks}")

    return track_arn


def get_track_name_from_arn(track_arn: str) -> str:
    """Get track name from ARN.

    Parameters
    ----------
    track_arn : str
        Track ARN

    Returns
    -------
    str
        Track name

    Raises
    ------
    ValueError
        If ARN format is invalid
    """
    try:
        track_name = track_arn.split("/")[-1]
        return track_name
    except (IndexError, AttributeError):
        raise ValueError(f"Invalid track ARN format: {track_arn}")


def get_tracks_by_type(track_type: str) -> List[TrackType]:
    """Get tracks by category type.

    Parameters
    ----------
    track_type : str
        Type category: 'official', 'championship', 'professional', 'open', 'special'

    Returns
    -------
    List[TrackType]
        List of tracks in the specified category
    """
    type_mapping = {
        "official": [
            TrackType.REINVENT_BASE,
            TrackType.OVAL_TRACK,
            TrackType.BOWTIE_TRACK,
            TrackType.SPEED_TRACK,
            TrackType.TIME_TRIAL_TRACK,
            TrackType.HEAD_TO_HEAD_TRACK,
        ],
        "championship": TrackType.get_championship_tracks(),
        "professional": TrackType.get_professional_tracks(),
        "open": [TrackType.ARCTIC_OPEN, TrackType.DUBAI_OPEN, TrackType.PENBAY_OPEN, TrackType.HAMPTION_OPEN],
        "special": [TrackType.SINGAPORE_F1, TrackType.MEXICO_TRACK, TrackType.NEW_YORK_EVAL, TrackType.ALBERT],
    }

    tracks = type_mapping.get(track_type.lower())
    if not tracks:
        available_types = list(type_mapping.keys())
        raise ValueError(f"Track type '{track_type}' not found. Available types: {available_types}")

    return tracks
