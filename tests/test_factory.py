from docquery_ingestion.factory import build_ingestor
from docquery_ingestion.ingestor import GLMOCRIngestor


def test_build_ingestor_from_the_documented_config_shape() -> None:
    config = {
        "ingestion": {
            "max_file_size_mb": 50,
            "retry_max_attempts": 5,
            "batch_size": 4,
        }
    }

    ingestor = build_ingestor(config, ocr_service_url="http://ocr:8080")

    assert isinstance(ingestor, GLMOCRIngestor)
    assert ingestor._config.max_file_size_mb == 50
    assert ingestor._config.retry_max_attempts == 5
    assert ingestor._config.batch_size == 4


def test_build_ingestor_applies_sensible_defaults() -> None:
    ingestor = build_ingestor({"ingestion": {}}, ocr_service_url="http://ocr:8080")

    assert ingestor._config.model == "ggml-org/GLM-OCR-GGUF:Q8_0"
    assert ingestor._config.prompt == "OCR"
