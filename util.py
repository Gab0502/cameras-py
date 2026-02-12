from collections import Counter
import string
import re
from paddleocr import PaddleOCR

ocr = PaddleOCR(
    use_angle_cls=True,  
    lang='en'            
)
def normalize_chars(text: str) -> str:
    corrections = {
        "0": "O", "O": "0",
        "1": "I", "I": "1",
        "5": "S", "S": "5",
        "8": "B", "B": "8",
    }
    result = []
    for i, ch in enumerate(text):
        if i < 3 and ch in corrections and corrections[ch].isalpha():
            result.append(corrections[ch])
        elif i >= 3 and ch in corrections and corrections[ch].isdigit():
            result.append(corrections[ch])
        else:
            result.append(ch)
    return "".join(result)
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}


def write_csv(results, output_path):
    """
    Write the results to a CSV file.

    Args:
        results (dict): Dictionary containing the results.
        output_path (str): Path to the output CSV file.
    """
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox',
                                                'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                'license_number_score'))

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                print(results[frame_nmr][car_id])
                if 'car' in results[frame_nmr][car_id].keys() and \
                   'license_plate' in results[frame_nmr][car_id].keys() and \
                   'text' in results[frame_nmr][car_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr,
                                                            car_id,
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['car']['bbox'][0],
                                                                results[frame_nmr][car_id]['car']['bbox'][1],
                                                                results[frame_nmr][car_id]['car']['bbox'][2],
                                                                results[frame_nmr][car_id]['car']['bbox'][3]),
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][0],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][1],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][2],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][3]),
                                                            results[frame_nmr][car_id]['license_plate']['bbox_score'],
                                                            results[frame_nmr][car_id]['license_plate']['text'],
                                                            results[frame_nmr][car_id]['license_plate']['text_score'])
                            )
        f.close()


def license_complies_format(text):
    """
    Check if the license plate text complies with the required format.

    Args:
        text (str): License plate text.

    Returns:
        bool: True if the license plate complies with the format, False otherwise.
    """
    if len(text) != 7:
        return False

    if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
       (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
       (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in dict_char_to_int.keys()) and \
       (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and \
       (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
       (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and \
       (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys()):
        return True
    else:
        return False


def format_license(text):
    """
    Format the license plate text by converting characters using the mapping dictionaries.

    Args:
        text (str): License plate text.

    Returns:
        str: Formatted license plate text.
    """
    license_plate_ = ''
    mapping = {0: dict_int_to_char, 1: dict_int_to_char, 4: dict_int_to_char, 5: dict_int_to_char, 6: dict_int_to_char,
               2: dict_char_to_int, 3: dict_char_to_int}
    for j in [0, 1, 2, 3, 4, 5, 6]:
        if text[j] in mapping[j].keys():
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]

    return license_plate_


import cv2
from datetime import datetime

def read_license_plate(license_plate_crop):
    """
    OCR robusto usando PaddleOCR - versão corrigida para a estrutura atual
    """
    try:
        result = ocr.ocr(license_plate_crop)
        

        if not result or not isinstance(result, list) or len(result) == 0:
            return None, 0.0
        
        result_dict = result[0]
        
        if 'rec_texts' in result_dict and 'rec_scores' in result_dict:
            rec_texts = result_dict['rec_texts']
            rec_scores = result_dict['rec_scores']
            

            
            best_text = None
            best_score = 0.0
            
            for text, score in zip(rec_texts, rec_scores):
                if text and text.strip():  
                    text = re.sub(r'[^A-Z0-9]', '', text.upper())
                    text = normalize_chars(text)
                    
                    if 4 <= len(text) <= 8:  
                        if score > best_score:
                            best_text = text
                            best_score = score
            
            if best_text:
                return best_text, best_score
            else:
                return None, 0.0
        else:
            print("Estrutura de resultado inesperada")
            return None, 0.0
            
    except Exception as e:
        print(f"Erro no OCR: {e}")
        import traceback
        traceback.print_exc()
        return None, 0.0

def read_license_plate_robust(license_plate_crop):
    """
    Versão mais robusta que tenta diferentes formas de extrair o texto
    """
    try:
        result = ocr.ocr(license_plate_crop)
        
        if not result:
            return None, 0.0
            
        result_dict = result[0]
        
        text_keys = ['rec_texts', 'text', 'result', 'detection_texts']
        score_keys = ['rec_scores', 'scores', 'confidence']
        
        for text_key, score_key in zip(text_keys, score_keys):
            if text_key in result_dict and score_key in result_dict:
                texts = result_dict[text_key]
                scores = result_dict[score_key]
                
                if texts and scores and len(texts) == len(scores):
                    best_text = None
                    best_score = 0.0
                    
                    for text, score in zip(texts, scores):
                        if text and str(text).strip():
                            text = re.sub(r'[^A-Z0-9]', '', str(text).upper())
                            text = normalize_chars(text)
                            
                            if 4 <= len(text) <= 8 and score > best_score:
                                best_text = text
                                best_score = score
                    
                    if best_text:
                        return best_text, best_score
        
        return search_text_in_structure(result_dict)
        
    except Exception as e:
        print(f"Erro no OCR robusto: {e}")
        return None, 0.0

def search_text_in_structure(obj, depth=0, max_depth=3):
    """
    Função recursiva para buscar texto em qualquer parte da estrutura
    """
    if depth > max_depth:
        return None, 0.0
        
    best_text = None
    best_score = 0.0
    
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == 'text' and isinstance(value, str) and value.strip():
                text = re.sub(r'[^A-Z0-9]', '', value.upper())
                text = normalize_chars(text)
                if 4 <= len(text) <= 8:
                    score = find_associated_score(obj, key)
                    if score > best_score:
                        best_text = text
                        best_score = score
            
            text, score = search_text_in_structure(value, depth + 1, max_depth)
            if text and score > best_score:
                best_text = text
                best_score = score
                
    elif isinstance(obj, list):
        for item in obj:
            text, score = search_text_in_structure(item, depth + 1, max_depth)
            if text and score > best_score:
                best_text = text
                best_score = score
                
    return best_text, best_score

def find_associated_score(obj, text_key):
    """
    Tenta encontrar um score associado ao texto
    """
    score_keys = ['score', 'confidence', 'rec_score']
    for score_key in score_keys:
        if score_key in obj:
            score = obj[score_key]
            if isinstance(score, (int, float)) and 0 <= score <= 1:
                return score
    return 0.0


def get_car(license_plate, vehicle_track_ids):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.

    Args:
        license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
        vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.

    Returns:
        tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
    """
    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1
