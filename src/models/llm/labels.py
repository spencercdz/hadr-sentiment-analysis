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
        'other_aid': get_other_aid_labels(),
        'infrastructure_related': get_infrastructure_related_labels(),
        'transport': get_transport_labels(),
        'buildings': get_buildings_labels(),
        'electricity': get_electricity_labels(),
        'tools': get_tools_labels(),
        'hospitals': get_hospitals_labels(),
        'shops': get_shops_labels(),
        'aid_centers': get_aid_centers_labels(),
        'other_infrastructure': get_other_infrastructure_labels(),
        'weather_related': get_weather_related_labels(),
        'floods': get_floods_labels(),
        'storm': get_storm_labels(),
        'fire': get_fire_labels(),
        'earthquake': get_earthquake_labels(),
        'cold': get_cold_labels(),
        'other_weather': get_other_weather_labels(),
        'direct_report': get_direct_report_labels(),
    }

def get_genre_labels() -> Dict[int, str]:
    """
    Get mappings of genre label ids to human-readable labels.
    
    Returns:
        Dict[int, str]:
            - 0: 'direct' (direct messages)
            - 1: 'news' (news stories or bulletins)
            - 2: 'social media' (social posting)
    """
    return {
        0: 'direct',
        1: 'news',
        2: 'social media'
    }

def get_related_labels() -> Dict[int, str]:
    """
    Get mappings of related label ids to human-readable labels.
    
    Returns:
        Dict[int, str]:
            - 0: 'no' (not related)
            - 1: 'yes' (related)
            - 2: 'maybe' (possibly related)
    """
    return {
        0: 'no',
        1: 'yes',
        2: 'maybe'
    }

def get_request_labels() -> Dict[int, str]:
    """
    Get mappings of request label ids to human-readable labels.
    
    Returns:
        Dict[int, str]:
            - 0: 'no' (no request)
            - 1: 'yes' (contains request)
    """
    return {
        0: 'no',
        1: 'yes'
    }

def get_offer_labels() -> Dict[int, str]:
    """
    Get mappings of offer label ids to human-readable labels.
    
    Returns:
        Dict[int, str]:
            - 0: 'no' (no offer)
            - 1: 'yes' (contains offer)
    """
    return {
        0: 'no',
        1: 'yes'
    }

def get_aid_related_labels() -> Dict[int, str]:
    """
    Get mappings of aid-related label ids to human-readable labels.
    
    Returns:
        Dict[int, str]:
            - 0: 'no' (not aid related)
            - 1: 'yes' (aid related)
    """
    return {
        0: 'no',
        1: 'yes'
    }

def get_medical_help_labels() -> Dict[int, str]:
    """
    Get mappings of medical help label ids to human-readable labels.
    
    Returns:
        Dict[int, str]:
            - 0: 'no' (no medical help)
            - 1: 'yes' (concerns medical help)
    """
    return {
        0: 'no',
        1: 'yes'
    }

def get_medical_products_labels() -> Dict[int, str]:
    """
    Get mappings of medical products label ids to human-readable labels.
    
    Returns:
        Dict[int, str]:
            - 0: 'no' (no medical products)
            - 1: 'yes' (concerns medical products)
    """
    return {
        0: 'no',
        1: 'yes'
    }

def get_search_and_rescue_labels() -> Dict[int, str]:
    """
    Get mappings of search and rescue label ids to human-readable labels.
    
    Returns:
        Dict[int, str]:
            - 0: 'no' (no search and rescue)
            - 1: 'yes' (concerns search and rescue)
    """
    return {
        0: 'no',
        1: 'yes'
    }

def get_security_labels() -> Dict[int, str]:
    """
    Get mappings of security label ids to human-readable labels.
    
    Returns:
        Dict[int, str]:
            - 0: 'no' (no security)
            - 1: 'yes' (concerns security)
    """
    return {
        0: 'no',
        1: 'yes'
    }

def get_military_labels() -> Dict[int, str]:
    """
    Get mappings of military label ids to human-readable labels.
    
    Returns:
        Dict[int, str]:
            - 0: 'no' (no military)
            - 1: 'yes' (concerns military)
    """
    return {
        0: 'no',
        1: 'yes'
    }

def get_child_alone_labels() -> Dict[int, str]:
    """
    Get mappings of child alone label ids to human-readable labels.
    
    Returns:
        Dict[int, str]:
            - 0: 'no' (no child alone)
            - 1: 'yes' (mentions child alone)
    """
    return {
        0: 'no',
        1: 'yes'
    }

def get_water_labels() -> Dict[int, str]:
    """
    Get mappings of water label ids to human-readable labels.
    
    Returns:
        Dict[int, str]:
            - 0: 'no' (no water)
            - 1: 'yes' (concerns water)
    """
    return {
        0: 'no',
        1: 'yes'
    }

def get_food_labels() -> Dict[int, str]:
    """
    Get mappings of food label ids to human-readable labels.
    
    Returns:
        Dict[int, str]:
            - 0: 'no' (no food)
            - 1: 'yes' (concerns food)
    """
    return {
        0: 'no',
        1: 'yes'
    }

def get_shelter_labels() -> Dict[int, str]:
    """
    Get mappings of shelter label ids to human-readable labels.
    
    Returns:
        Dict[int, str]:
            - 0: 'no' (no shelter)
            - 1: 'yes' (concerns shelter)
    """
    return {
        0: 'no',
        1: 'yes'
    }

def get_clothing_labels() -> Dict[int, str]:
    """
    Get mappings of clothing label ids to human-readable labels.
    
    Returns:
        Dict[int, str]:
            - 0: 'no' (no clothing)
            - 1: 'yes' (concerns clothing)
    """
    return {
        0: 'no',
        1: 'yes'
    }

def get_money_labels() -> Dict[int, str]:
    """
    Get mappings of money label ids to human-readable labels.
    
    Returns:
        Dict[int, str]:
            - 0: 'no' (no money)
            - 1: 'yes' (concerns money)
    """
    return {
        0: 'no',
        1: 'yes'
    }

def get_missing_people_labels() -> Dict[int, str]:
    """
    Get mappings of missing people label ids to human-readable labels.
    
    Returns:
        Dict[int, str]:
            - 0: 'no' (no missing people)
            - 1: 'yes' (indicates missing people)
    """
    return {
        0: 'no',
        1: 'yes'
    }

def get_refugees_labels() -> Dict[int, str]:
    """
    Get mappings of refugees label ids to human-readable labels.
    
    Returns:
        Dict[int, str]:
            - 0: 'no' (no refugees)
            - 1: 'yes' (concerns refugees)
    """
    return {
        0: 'no',
        1: 'yes'
    }

def get_death_labels() -> Dict[int, str]:
    """
    Get mappings of death label ids to human-readable labels.
    
    Returns:
        Dict[int, str]:
            - 0: 'no' (no death)
            - 1: 'yes' (implies death)
    """
    return {
        0: 'no',
        1: 'yes'
    }

def get_other_aid_labels() -> Dict[int, str]:
    """
    Get mappings of other aid label ids to human-readable labels.
    
    Returns:
        Dict[int, str]:
            - 0: 'no' (no other aid)
            - 1: 'yes' (other aid needed)
    """
    return {
        0: 'no',
        1: 'yes'
    }

def get_infrastructure_related_labels() -> Dict[int, str]:
    """
    Get mappings of infrastructure-related label ids to human-readable labels.
    
    Returns:
        Dict[int, str]:
            - 0: 'no' (no infrastructure)
            - 1: 'yes' (concerns infrastructure)
    """
    return {
        0: 'no',
        1: 'yes'
    }

def get_transport_labels() -> Dict[int, str]:
    """
    Get mappings of transport label ids to human-readable labels.
    
    Returns:
        Dict[int, str]:
            - 0: 'no' (no transport)
            - 1: 'yes' (concerns transport)
    """
    return {
        0: 'no',
        1: 'yes'
    }

def get_buildings_labels() -> Dict[int, str]:
    """
    Get mappings of buildings label ids to human-readable labels.
    
    Returns:
        Dict[int, str]:
            - 0: 'no' (no buildings)
            - 1: 'yes' (concerns buildings)
    """
    return {
        0: 'no',
        1: 'yes'
    }

def get_electricity_labels() -> Dict[int, str]:
    """
    Get mappings of electricity label ids to human-readable labels.
    
    Returns:
        Dict[int, str]:
            - 0: 'no' (no electricity)
            - 1: 'yes' (concerns electricity)
    """
    return {
        0: 'no',
        1: 'yes'
    }

def get_tools_labels() -> Dict[int, str]:
    """
    Get mappings of tools label ids to human-readable labels.
    
    Returns:
        Dict[int, str]:
            - 0: 'no' (no tools)
            - 1: 'yes' (concerns tools)
    """
    return {
        0: 'no',
        1: 'yes'
    }

def get_hospitals_labels() -> Dict[int, str]:
    """
    Get mappings of hospitals label ids to human-readable labels.
    
    Returns:
        Dict[int, str]:
            - 0: 'no' (no hospitals)
            - 1: 'yes' (concerns hospitals)
    """
    return {
        0: 'no',
        1: 'yes'
    }

def get_shops_labels() -> Dict[int, str]:
    """
    Get mappings of shops label ids to human-readable labels.
    
    Returns:
        Dict[int, str]:
            - 0: 'no' (no shops)
            - 1: 'yes' (concerns shops)
    """
    return {
        0: 'no',
        1: 'yes'
    }

def get_aid_centers_labels() -> Dict[int, str]:
    """
    Get mappings of aid centers label ids to human-readable labels.
    
    Returns:
        Dict[int, str]:
            - 0: 'no' (no aid centers)
            - 1: 'yes' (concerns aid centers)
    """
    return {
        0: 'no',
        1: 'yes'
    }

def get_other_infrastructure_labels() -> Dict[int, str]:
    """
    Get mappings of other infrastructure label ids to human-readable labels.
    
    Returns:
        Dict[int, str]:
            - 0: 'no' (no other infrastructure)
            - 1: 'yes' (concerns other infrastructure)
    """
    return {
        0: 'no',
        1: 'yes'
    }

def get_weather_related_labels() -> Dict[int, str]:
    """
    Get mappings of weather-related label ids to human-readable labels.
    
    Returns:
        Dict[int, str]:
            - 0: 'no' (no weather)
            - 1: 'yes' (concerns weather)
    """
    return {
        0: 'no',
        1: 'yes'
    }

def get_floods_labels() -> Dict[int, str]:
    """
    Get mappings of floods label ids to human-readable labels.
    
    Returns:
        Dict[int, str]:
            - 0: 'no' (no flood)
            - 1: 'yes' (indicates flood)
    """
    return {
        0: 'no',
        1: 'yes'
    }

def get_storm_labels() -> Dict[int, str]:
    """
    Get mappings of storm label ids to human-readable labels.
    
    Returns:
        Dict[int, str]:
            - 0: 'no' (no storm)
            - 1: 'yes' (indicates storm)
    """
    return {
        0: 'no',
        1: 'yes'
    }

def get_fire_labels() -> Dict[int, str]:
    """
    Get mappings of fire label ids to human-readable labels.
    
    Returns:
        Dict[int, str]:
            - 0: 'no' (no fire)
            - 1: 'yes' (indicates fire)
    """
    return {
        0: 'no',
        1: 'yes'
    }

def get_earthquake_labels() -> Dict[int, str]:
    """
    Get mappings of earthquake label ids to human-readable labels.
    
    Returns:
        Dict[int, str]:
            - 0: 'no' (no earthquake)
            - 1: 'yes' (indicates earthquake)
    """
    return {
        0: 'no',
        1: 'yes'
    }

def get_cold_labels() -> Dict[int, str]:
    """
    Get mappings of cold label ids to human-readable labels.
    
    Returns:
        Dict[int, str]:
            - 0: 'no' (no cold)
            - 1: 'yes' (indicates cold)
    """
    return {
        0: 'no',
        1: 'yes'
    }

def get_other_weather_labels() -> Dict[int, str]:
    """
    Get mappings of other weather label ids to human-readable labels.
    
    Returns:
        Dict[int, str]:
            - 0: 'no' (no other weather)
            - 1: 'yes' (indicates other weather issues)
    """
    return {
        0: 'no',
        1: 'yes'
    }

def get_direct_report_labels() -> Dict[int, str]:
    """
    Get mappings of direct report label ids to human-readable labels.
    
    Returns:
        Dict[int, str]:
            - 0: 'no' (not a direct report)
            - 1: 'yes' (is a direct report)
    """
    return {
        0: 'no',
        1: 'yes'
    }

def get_sentiment_labels() -> Dict[int, str]:
    """
    Get mappings of sentiment label ids to human-readable labels.
    
    Returns:
        Dict[int, str]:
            - 0: 'negative' (negative sentiment)
            - 1: 'neutral' (neutral sentiment)
            - 2: 'positive' (positive sentiment)
    """
    return {
        0: 'negative',
        1: 'neutral',
        2: 'positive'
    }