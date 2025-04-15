from typing import Dict

def get_all_labels() -> Dict[str, Dict[int, str]]:
    """
    Get mappings of all task labels to their human-readable labels.
    
    Returns:
        Dict[str, Dict[int, str]]: A dictionary mapping each task (as a key)
        to its dictionary of label ids and labels.
    """
    return {
        'sentiment': get_sentiment_labels(),
        'genre': get_genre_labels(),
        'related': get_related_labels(),
        'request': get_request_labels(),
        'offer': get_offer_labels(),
        'aid_related': get_aid_related_labels(),
        'medical_help': get_medical_help_labels(),
        'medical_products': get_medical_products_labels(),
        'search_and_rescue': get_search_and_rescue_labels(),
        'security': get_security_labels(),
        'military': get_military_labels(),
        'child_alone': get_child_alone_labels(),
        'water': get_water_labels(),
        'food': get_food_labels(),
        'shelter': get_shelter_labels(),
        'clothing': get_clothing_labels(),
        'money': get_money_labels(),
        'missing_people': get_missing_people_labels(),
        'refugees': get_refugees_labels(),
        'death': get_death_labels(),
        'other aid': get_other_aid_labels(),
        'infrastructure_related': get_infrastructure_related_labels(),
        'other_infrastructure': get_other_infrastructure_labels(),
        'weather_related': get_weather_related_labels(),
        'floods': get_floods_labels(),
        'storm': get_storm_labels(),
        'fire': get_fire_labels(),
        'earthquake': get_earthquake_labels(),
        'cold': get_cold_labels(),
        'other_weather': get_other_weather_labels(),
        'direct_report': get_direct_report_labels()
    }

def get_sentiment_labels() -> Dict[int, str]:
    """
    Get mappings of sentiment label ids to human-readable labels.
    
    Returns:
        Dict[int, str]: 
            - 0: 'non-negative' (sentiment is not negative)
            - 1: 'negative' (negative sentiment)
    """
    return {
        0: 'non-negative',
        1: 'negative'
    }

def get_genre_labels() -> Dict[int, str]:
    """
    Get mappings of genre label ids to human-readable labels.
    
    Returns:
        Dict[int, str]:
            - 0: 'news'
            - 1: 'direct'
            - 2: 'social media'
    """
    return {
        0: 'news',
        1: 'direct',
        2: 'social media'
    }

def get_related_labels() -> Dict[int, str]:
    """
    Get mappings of related label ids to human-readable labels.
    
    Returns:
        Dict[int, str]:
            - 0: 'not related'
            - 1: 'related'
            - 2: 'maybe'
    """
    return {
        0: 'not related',
        1: 'related',
        2: 'maybe'
    }

def get_request_labels() -> Dict[int, str]:
    """
    Get mappings of request label ids to human-readable labels.
    
    Returns:
        Dict[int, str]:
            - 0: 'not a request'
            - 1: 'request'
    """
    return {
        0: 'not a request',
        1: 'request'
    }

def get_offer_labels() -> Dict[int, str]:
    """
    Get mappings of offer label ids to human-readable labels.
    
    Returns:
        Dict[int, str]:
            - 0: 'not offer'
            - 1: 'offer'
    """
    return {
        0: 'not offer',
        1: 'offer'
    }

def get_aid_related_labels() -> Dict[int, str]:
    """
    Get mappings of aid-related label ids to human-readable labels.
    
    Returns:
        Dict[int, str]:
            - 0: 'not aid related'
            - 1: 'aid related'
    """
    return {
        0: 'not aid related',
        1: 'aid related'
    }

def get_medical_help_labels() -> Dict[int, str]:
    """
    Get mappings of medical help label ids to human-readable labels.
    
    Returns:
        Dict[int, str]:
            - 0: 'not medical help'
            - 1: 'medical help'
    """
    return {
        0: 'not medical help',
        1: 'medical help'
    }

def get_medical_products_labels() -> Dict[int, str]:
    """
    Get mappings of medical products label ids to human-readable labels.
    
    Returns:
        Dict[int, str]:
            - 0: 'not medical products'
            - 1: 'medical products'
    """
    return {
        0: 'not medical products',
        1: 'medical products'
    }

def get_search_and_rescue_labels() -> Dict[int, str]:
    """
    Get mappings of search and rescue label ids to human-readable labels.
    
    Returns:
        Dict[int, str]:
            - 0: 'not search and rescue'
            - 1: 'search and rescue'
    """
    return {
        0: 'not search and rescue',
        1: 'search and rescue'
    }

def get_security_labels() -> Dict[int, str]:
    """
    Get mappings of security label ids to human-readable labels.
    
    Returns:
        Dict[int, str]:
            - 0: 'not security'
            - 1: 'security'
    """
    return {
        0: 'not security',
        1: 'security'
    }

def get_military_labels() -> Dict[int, str]:
    """
    Get mappings of military label ids to human-readable labels.
    
    Returns:
        Dict[int, str]:
            - 0: 'not military'
            - 1: 'military'
    """
    return {
        0: 'not military',
        1: 'military'
    }

def get_child_alone_labels() -> Dict[int, str]:
    """
    Get mappings of child alone label ids to human-readable labels.
    
    Returns:
        Dict[int, str]:
            - 0: 'not child alone'
            - 1: 'child alone'
    """
    return {
        0: 'not child alone',
        1: 'child alone'
    }

def get_water_labels() -> Dict[int, str]:
    """
    Get mappings of water label ids to human-readable labels.
    
    Returns:
        Dict[int, str]:
            - 0: 'not water'
            - 1: 'water'
    """
    return {
        0: 'not water',
        1: 'water'
    }

def get_food_labels() -> Dict[int, str]:
    """
    Get mappings of food label ids to human-readable labels.
    
    Returns:
        Dict[int, str]:
            - 0: 'not food'
            - 1: 'food'
    """
    return {
        0: 'not food',
        1: 'food'
    }

def get_shelter_labels() -> Dict[int, str]:
    """
    Get mappings of shelter label ids to human-readable labels.
    
    Returns:
        Dict[int, str]:
            - 0: 'not shelter'
            - 1: 'shelter'
    """
    return {
        0: 'not shelter',
        1: 'shelter'
    }

def get_clothing_labels() -> Dict[int, str]:
    """
    Get mappings of clothing label ids to human-readable labels.
    
    Returns:
        Dict[int, str]:
            - 0: 'not clothing'
            - 1: 'clothing'
    """
    return {
        0: 'not clothing',
        1: 'clothing'
    }

def get_money_labels() -> Dict[int, str]:
    """
    Get mappings of money label ids to human-readable labels.
    
    Returns:
        Dict[int, str]:
            - 0: 'not money'
            - 1: 'money'
    """
    return {
        0: 'not money',
        1: 'money'
    }

def get_missing_people_labels() -> Dict[int, str]:
    """
    Get mappings of missing people label ids to human-readable labels.
    
    Returns:
        Dict[int, str]:
            - 0: 'not missing people'
            - 1: 'missing people'
    """
    return {
        0: 'not missing people',
        1: 'missing people'
    }

def get_refugees_labels() -> Dict[int, str]:
    """
    Get mappings of refugees label ids to human-readable labels.
    
    Returns:
        Dict[int, str]:
            - 0: 'not refugees'
            - 1: 'refugees'
    """
    return {
        0: 'not refugees',
        1: 'refugees'
    }

def get_death_labels() -> Dict[int, str]:
    """
    Get mappings of death label ids to human-readable labels.
    
    Returns:
        Dict[int, str]:
            - 0: 'not death'
            - 1: 'death'
    """
    return {
        0: 'not death',
        1: 'death'
    }

def get_other_aid_labels() -> Dict[int, str]:
    """
    Get mappings of other aid label ids to human-readable labels.
    
    Returns:
        Dict[int, str]:
            - 0: 'not other aid'
            - 1: 'other aid'
    """
    return {
        0: 'not other aid',
        1: 'other aid'
    }

def get_infrastructure_related_labels() -> Dict[int, str]:
    """
    Get mappings of infrastructure-related label ids to human-readable labels.
    
    Returns:
        Dict[int, str]:
            - 0: 'not infrastructure related'
            - 1: 'infrastructure related'
    """
    return {
        0: 'not infrastructure related',
        1: 'infrastructure related'
    }

def get_other_infrastructure_labels() -> Dict[int, str]:
    """
    Get mappings of other infrastructure label ids to human-readable labels.
    
    Returns:
        Dict[int, str]:
            - 0: 'not other infrastructure'
            - 1: 'other infrastructure'
    """
    return {
        0: 'not other infrastructure',
        1: 'other infrastructure'
    }

def get_weather_related_labels() -> Dict[int, str]:
    """
    Get mappings of weather-related label ids to human-readable labels.
    
    Returns:
        Dict[int, str]:
            - 0: 'not weather'
            - 1: 'weather'
    """
    return {
        0: 'not weather',
        1: 'weather'
    }

def get_floods_labels() -> Dict[int, str]:
    """
    Get mappings of flood label ids to human-readable labels.
    
    Returns:
        Dict[int, str]:
            - 0: 'not flood'
            - 1: 'flood'
    """
    return {
        0: 'not flood',
        1: 'flood'
    }

def get_storm_labels() -> Dict[int, str]:
    """
    Get mappings of storm label ids to human-readable labels.
    
    Returns:
        Dict[int, str]:
            - 0: 'not storm'
            - 1: 'storm'
    """
    return {
        0: 'not storm',
        1: 'storm'
    }

def get_fire_labels() -> Dict[int, str]:
    """
    Get mappings of fire label ids to human-readable labels.
    
    Returns:
        Dict[int, str]:
            - 0: 'not fire'
            - 1: 'fire'
    """
    return {
        0: 'not fire',
        1: 'fire'
    }

def get_earthquake_labels() -> Dict[int, str]:
    """
    Get mappings of earthquake label ids to human-readable labels.
    
    Returns:
        Dict[int, str]:
            - 0: 'not earthquake'
            - 1: 'earthquake'
    """
    return {
        0: 'not earthquake',
        1: 'earthquake'
    }

def get_cold_labels() -> Dict[int, str]:
    """
    Get mappings of cold label ids to human-readable labels.
    
    Returns:
        Dict[int, str]:
            - 0: 'not cold'
            - 1: 'cold'
    """
    return {
        0: 'not cold',
        1: 'cold'
    }

def get_other_weather_labels() -> Dict[int, str]:
    """
    Get mappings of other weather label ids to human-readable labels.
    
    Returns:
        Dict[int, str]:
            - 0: 'not other weather'
            - 1: 'other weather'
    """
    return {
        0: 'not other weather',
        1: 'other weather'
    }

def get_direct_report_labels() -> Dict[int, str]:
    """
    Get mappings of direct report label ids to human-readable labels.
    
    Returns:
        Dict[int, str]:
            - 0: 'not a direct report'
            - 1: 'direct report'
    """
    return {
        0: 'not a direct report',
        1: 'direct report'
    }





""" OLD LABELS """

# REMOVE: Get sentiment labels
def get_sentiment_labels() -> Dict[int, str]:
    """
    Get mappings of sentiment label ids to human-readable labels.

    Returns:
        Dict[int, str]: A dictionary mapping label ids to labels.
    """
    return {
        0: 'non-negative',
        1: 'negative'
    }

# REMOVE: Get sentiment labels
def get_event_type_labels() -> Dict[int, str]:
    """
    Get mappings of event type label ids to human-readable labels.
    """
    return {
        0: 'unknown',
        1: 'storm',
        2: 'flood',
        3: 'earthquake',
        4: 'fire',
        5: 'meteor',
        6: 'volcano',
        7: 'landslide',
        8: 'haze',
    }

# REMOVE: Get sentiment labels
def get_event_type_detail_labels() -> Dict[int, str]:
    """
    Get mappings of event type detail label ids to human-readable labels.
    """
    return {
        0: 'unknown',
        1: 'avalanche',
        2: 'blizzard',
        3: 'bush_fire',
        4: 'cyclone',
        5: 'dust_storm',
        6: 'earthquake',
        7: 'flood',
        8: 'forest_fire',
        9: 'haze',
        10: 'hurricane',
        11: 'landslide',
        12: 'meteor',
        13: 'storm',
        14: 'tornado',
        15: 'tsunami',
        16: 'typhoon',
        17: 'volcano',
        18: 'wildfire'
    }

# REMOVE: Get sentiment labels
def get_label_labels() -> Dict[int, str]:
    """
    Get mappings of label label ids to human-readable labels.
    """
    return {
        0: 'irrelvant',
        1: 'dont_know'
    }