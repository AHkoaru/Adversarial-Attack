from PIL import Image
import numpy as np
import os
import json


def convert_to_train_id(label_array):
    """
    Cityscapes의 원본 레이블을 학습에 사용되는 레이블로 변환합니다.
    """
    mapping = {
        7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7,
        21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
        28: 15, 31: 16, 32: 17, 33: 18
    }
    return np.vectorize(lambda x: mapping.get(x, 255))(label_array)

def label_to_train_id(gt_mask):
    gt = np.array(gt_mask) if isinstance(gt_mask, Image.Image) else np.array(gt_mask)
    gt = convert_to_train_id(gt)
    return gt

def save_results(results, filename):
    """
    결과를 JSON 파일에 저장합니다.
    
    Args:
        results: 저장할 결과 딕셔너리
        filename: 저장할 파일 이름
    """
    
    Result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Result")
    if not os.path.exists(Result_dir):
        os.makedirs(Result_dir)
    file_path = os.path.join(Result_dir, filename)
    
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            # 기존 파일이 리스트 형식이 아닐 경우 리스트로 감싸줍니다.
            existing_results = json.load(f)
            if not isinstance(existing_results, list):
                existing_results = [existing_results]
    else:
        existing_results = []
    
    existing_results.append(results)
    
    with open(file_path, "w") as f:
        json.dump(existing_results, f, ensure_ascii=False, indent=4)
    print(f"결과가 '{file_path}' 파일에 추가 저장되었습니다.")