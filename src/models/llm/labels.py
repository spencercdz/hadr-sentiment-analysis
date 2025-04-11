from typing import Dict

def get_all_labels() -> Dict[int, str]:
    """Get mappings of all label ids to human-readable labels.
    
    Returns:
        Dict[int, str]: A dictionary mapping label ids to labels.
    """
    return {
        **get_sentiment_labels(),
        **get_genre_labels(),
        **get_related_labels(),
        **get_request_labels(),
        **get_aid_related_labels(),
        **get_medical_help_labels(),
        **get_shelter_labels(),
        **get_child_alone_labels(), 
        **get_food_labels(),
        **get_water_labels(),
        **get_shelter_labels(),
        **get_clothing_labels(),
        **get_money_labels(),
        **get_missing_people_labels(),
        **get_refugees_labels(),
        **get_deaths_labels(),
        **get_weather_labels(),
        **get_flood_labels(),
        **get_storm_labels(),
        **get_fire_labels(),
        **get_earthquake_labels(),
        **get_cold_labels(),
        **get_other_weather_labels(),
        **get_direct_report_labels()
    }

def get_sentiment_labels() -> Dict[int, str]:
    """Get mappings of sentiment label ids to human-readable labels.
    
    Returns:
        Dict[int, str]: A dictionary mapping label ids to labels.
        - 0: 'non-negative' - Indicates that the sentiment of the message is not negative.
        - 1: 'negative' - Indicates that the sentiment of the message is negative.
    """
    return {
        0: 'non-negative',
        1: 'negative'
    }

def get_genre_labels() -> Dict[int, str]:
    """Get mappings of genre label ids to human-readable labels.
    
    Returns:
        Dict[int, str]: A dictionary mapping label ids to labels.
        - 0: 'news' - Indicates that the message is a news article.
        - 1: 'direct' - Indicates that the message is a direct communication.
        - 2: 'social media' - Indicates that the message is from a social media platform.
    """
    return {
        0: 'news',
        1: 'direct',
        2: 'social media'
    }

def get_related_labels() -> Dict[int, str]:
    """Get mappings of related label ids to human-readable labels.
    
    Returns:
        Dict[int, str]: A dictionary mapping label ids to labels.
        - 0: 'not related' - Indicates that the message is not related to a disaster.
        - 1: 'related' - Indicates that the message is related to a disaster.
    """
    return {
        0: 'not related',
        1: 'related'
    }

def get_request_labels() -> Dict[int, str]:
    """Get mappings of request label ids to human-readable labels.
    
    Returns:
        Dict[int, str]: A dictionary mapping label ids to labels.
        - 0: 'not a request' - Indicates that the message does not contain a request.
        - 1: 'request' - Indicates that the message contains a request for assistance.
    """
    return {
        0: 'not a request',
        1: 'request'
    }

def get_aid_related_labels() -> Dict[int, str]:
    """Get mappings of aid-related label ids to human-readable labels.
    
    Returns:
        Dict[int, str]: A dictionary mapping label ids to labels.
        - 0: 'not aid related' - Indicates that the message is not related to aid.
        - 1: 'aid related' - Indicates that the message is related to aid.
    """
    return {
        0: 'not aid related',
        1: 'aid related'
    }

def get_medical_help_labels() -> Dict[int, str]:
    """Get mappings of medical help label ids to human-readable labels.
    
    Returns:
        Dict[int, str]: A dictionary mapping label ids to labels.
        - 0: 'not medical help' - Indicates that the message does not concern medical help.
        - 1: 'medical help' - Indicates that the message concerns medical help.
    """
    return {
        0: 'not medical help',
        1: 'medical help'
    }

def get_shelter_labels() -> Dict[int, str]:
    """Get mappings of shelter label ids to human-readable labels.
    
    Returns:
        Dict[int, str]: A dictionary mapping label ids to labels.
        - 0: 'not shelter' - Indicates that the message does not concern shelter.
        - 1: 'shelter' - Indicates that the message concerns shelter.
    """
    return {
        0: 'not shelter',
        1: 'shelter'
    }

def get_child_alone_labels() -> Dict[int, str]:
    """Get mappings of child alone label ids to human-readable labels.
    
    Returns:
        Dict[int, str]: A dictionary mapping label ids to labels.
        - 0: 'not child alone' - Indicates that the message does not mention a child alone.
        - 1: 'child alone' - Indicates that the message mentions a child alone.
    """
    return {
        0: 'not child alone',
        1: 'child alone'
    }

def get_food_labels() -> Dict[int, str]:
    """Get mappings of food label ids to human-readable labels.
    
    Returns:
        Dict[int, str]: A dictionary mapping label ids to labels.
        - 0: 'not food' - Indicates that the message does not concern food.
        - 1: 'food' - Indicates that the message concerns food.
    """
    return {
        0: 'not food',
        1: 'food'
    }

def get_water_labels() -> Dict[int, str]:
    """Get mappings of water label ids to human-readable labels.
    
    Returns:
        Dict[int, str]: A dictionary mapping label ids to labels.
        - 0: 'not water' - Indicates that the message does not concern water.
        - 1: 'water' - Indicates that the message concerns water.
    }
    """
    return {
        0: 'not water',
        1: 'water'
    }

def get_shelter_labels() -> Dict[int, str]:
    """Get mappings of shelter label ids to human-readable labels.
    
    Returns:
        Dict[int, str]: A dictionary mapping label ids to labels.
        - 0: 'not shelter' - Indicates that the message does not concern shelter.
        - 1: 'shelter' - Indicates that the message concerns shelter.
    """
    return {
        0: 'not shelter',
        1: 'shelter'
    }

def get_clothing_labels() -> Dict[int, str]:
    """Get mappings of clothing label ids to human-readable labels.
    
    Returns:
        Dict[int, str]: A dictionary mapping label ids to labels.
        - 0: 'not clothing' - Indicates that the message does not concern clothing.
        - 1: 'clothing' - Indicates that the message concerns clothing.
    """
    return {
        0: 'not clothing',
        1: 'clothing'
    }

def get_money_labels() -> Dict[int, str]:
    """Get mappings of money label ids to human-readable labels.
    
    Returns:
        Dict[int, str]: A dictionary mapping label ids to labels.
        - 0: 'not money' - Indicates that the message does not concern money.
        - 1: 'money' - Indicates that the message concerns money.
    """
    return {
        0: 'not money',
        1: 'money'
    }

def get_missing_people_labels() -> Dict[int, str]:
    """Get mappings of missing people label ids to human-readable labels.
    
    Returns:
        Dict[int, str]: A dictionary mapping label ids to labels.
        - 0: 'not missing people' - Indicates that the message does not indicate missing people.
        - 1: 'missing people' - Indicates that the message indicates missing people.
    """
    return {
        0: 'not missing people',
        1: 'missing people'
    }

def get_refugees_labels() -> Dict[int, str]:
    """Get mappings of refugees label ids to human-readable labels.
    
    Returns:
        Dict[int, str]: A dictionary mapping label ids to labels.
        - 0: 'not refugees' - Indicates that the message does not concern refugees.
        - 1: 'refugees' - Indicates that the message concerns refugees.
    """
    return {
        0: 'not refugees',
        1: 'refugees'
    }

def get_deaths_labels() -> Dict[int, str]:
    """Get mappings of deaths label ids to human-readable labels.
    
    Returns:
        Dict[int, str]: A dictionary mapping label ids to labels.
        - 0: 'not deaths' - Indicates that the message does not imply death.
        - 1: 'deaths' - Indicates that the message implies death.
    """
    return {
        0: 'not deaths',
        1: 'deaths'
    }

def get_weather_labels() -> Dict[int, str]:
    """Get mappings of weather label ids to human-readable labels.
    
    Returns:
        Dict[int, str]: A dictionary mapping label ids to labels.
        - 0: 'not weather' - Indicates that the message does not concern weather.
        - 1: 'weather' - Indicates that the message concerns weather.
    """
    return {
        0: 'not weather',
        1: 'weather'
    }

def get_flood_labels() -> Dict[int, str]:
    """Get mappings of flood label ids to human-readable labels.
    
    Returns:
        Dict[int, str]: A dictionary mapping label ids to labels.
        - 0: 'not flood' - Indicates that the message does not concern floods.
        - 1: 'flood' - Indicates that the message concerns floods.
    """
    return {
        0: 'not flood',
        1: 'flood'
    }

def get_storm_labels() -> Dict[int, str]:
    """Get mappings of storm label ids to human-readable labels.
    
    Returns:
        Dict[int, str]: A dictionary mapping label ids to labels.
        - 0: 'not storm' - Indicates that the message does not concern storms.
        - 1: 'storm' - Indicates that the message concerns storms.
    """
    return {
        0: 'not storm',
        1: 'storm'
    }

def get_fire_labels() -> Dict[int, str]:
    """Get mappings of fire label ids to human-readable labels.
    
    Returns:
        Dict[int, str]: A dictionary mapping label ids to labels.
        - 0: 'not fire' - Indicates that the message does not concern fires.
        - 1: 'fire' - Indicates that the message concerns fires.
    """
    return {
        0: 'not fire',
        1: 'fire'
    }

def get_earthquake_labels() -> Dict[int, str]:
    """Get mappings of earthquake label ids to human-readable labels.
    
    Returns:
        Dict[int, str]: A dictionary mapping label ids to labels.
        - 0: 'not earthquake' - Indicates that the message does not concern earthquakes.
        - 1: 'earthquake' - Indicates that the message concerns earthquakes.
    """
    return {
        0: 'not earthquake',
        1: 'earthquake'
    }

def get_cold_labels() -> Dict[int, str]:
    """Get mappings of cold label ids to human-readable labels.
    
    Returns:
        Dict[int, str]: A dictionary mapping label ids to labels.
        - 0: 'not cold' - Indicates that the message does not concern cold weather.
        - 1: 'cold' - Indicates that the message concerns cold weather.
    """
    return {
        0: 'not cold',
        1: 'cold'
    }

def get_other_weather_labels() -> Dict[int, str]:
    """Get mappings of other weather label ids to human-readable labels.
    
    Returns:
        Dict[int, str]: A dictionary mapping label ids to labels.
        - 0: 'not other weather' - Indicates that the message does not concern other weather issues.
        - 1: 'other weather' - Indicates that the message concerns other weather issues.
    """
    return {
        0: 'not other weather',
        1: 'other weather'
    }

def get_direct_report_labels() -> Dict[int, str]:
    """Get mappings of direct report label ids to human-readable labels.
    
    Returns:
        Dict[int, str]: A dictionary mapping label ids to labels.
        - 0: 'not a direct report' - Indicates that the message is not a direct report.
        - 1: 'direct report' - Indicates that the message is a direct report.
    """
    return {
        0: 'not a direct report',
        1: 'direct report'
    }   