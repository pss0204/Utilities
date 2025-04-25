#!/usr/bin/env python3
# filepath: /home/pss/etc/hf_data/gsk8k.py

"""
OpenAI의 GSM8K 데이터셋을 Hugging Face에서 로드하고 푸시하는 간단한 스크립트입니다.
"""

from datasets import load_dataset
import logging
import re

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_final_answer(answer_text):
    """
    "#### " 뒤의 숫자만 추출합니다.
    
    Args:
        answer_text: 원본 답변 텍스트
    
    Returns:
        추출된 최종 답변 (숫자만)
    """
    # "#### " 뒤의 숫자 패턴을 찾습니다
    match = re.search(r'####\s*(\d+\.?\d*)', answer_text)
    if match:
        return match.group(1)
    return answer_text  # 패턴이 없으면 원본 반환

def push_to_hub(dataset, dataset_id, config_name=None):
    """
    데이터셋을 Hugging Face Hub에 푸시합니다.
    
    Args:
        dataset: 푸시할 데이터셋
        dataset_id: 데이터셋 ID (예: "사용자명/데이터셋명")
        config_name: 설정 이름 (예: "gsm8k_test", "gsm8k_train" 등)
    """
    logger.info(f"데이터셋 푸시 중: {dataset_id}, 구성 이름: {config_name}")
    dataset.push_to_hub(dataset_id, config_name=config_name)
    logger.info(f"데이터셋 푸시 완료: {dataset_id}, 구성 이름: {config_name}")

if __name__ == "__main__":
    # GSM8K 데이터셋 로드
    logger.info("GSM8K 데이터셋 로드 중...")
    gsm8k_dataset = load_dataset("openai/gsm8k", "main", split="test")
    logger.info("GSM8K 데이터셋 로드 완료")
    
    # 데이터셋 기본 정보 출력
    logger.info(f"원본 데이터셋 구조: {gsm8k_dataset}")
    
    # 'question' 특성을 'problem'으로 리매핑
    logger.info("데이터셋 특성 리매핑: 'question' -> 'problem'")
    gsm8k_dataset = gsm8k_dataset.rename_column("question", "problem")
    
    # 'answer' 필드에서 최종 답변만 추출
    logger.info("'answer' 필드에서 최종 답변(#### 뒤의 숫자)만 추출")
    
    # 원본 answer 컬럼을 full_answer로 복사하고 새로운 answer 컬럼 생성
    gsm8k_dataset = gsm8k_dataset.add_column("full_answer", gsm8k_dataset["answer"])
    
    # answer 컬럼을 최종 답변만 추출한 값으로 업데이트
    gsm8k_dataset = gsm8k_dataset.map(
        lambda example: {"answer": extract_final_answer(example["answer"])}
    )
    
    # 샘플 데이터 출력하여 확인
    logger.info("변환된 데이터 샘플:")
    for i in range(min(3, len(gsm8k_dataset))):  # 처음 3개 샘플 확인
        logger.info(f"예제 {i+1}:")
        logger.info(f"문제: {gsm8k_dataset[i]['problem']}")
        logger.info(f"전체 답변: {gsm8k_dataset[i]['full_answer']}")
        logger.info(f"추출된 최종 답변: {gsm8k_dataset[i]['answer']}")
        logger.info("-" * 40)
    
    # 변경된 데이터셋 구조 출력
    logger.info(f"변경된 데이터셋 구조: {gsm8k_dataset}")
    
    # Hugging Face Hub에 푸시
    # pss0204/pss_sal 레포지토리에 구성 이름을 지정하여 푸시합니다
    config_name = "gsm8k_test"  # 원하는 구성 이름으로 변경하세요
    push_to_hub(gsm8k_dataset, "pss0204/pss_sal", config_name=config_name)


