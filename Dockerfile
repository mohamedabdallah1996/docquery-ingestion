FROM ghcr.io/ggml-org/llama.cpp:server-cuda

# LLAMA_CACHE is the documented cache dir for `-hf` downloads, but some
# versions honor HF_HOME/HF_HUB_CACHE instead (ggml-org/llama.cpp#18684) --
# set both so the volume below reliably catches the download either way.
ENV LLAMA_CACHE=/models
ENV HF_HOME=/models

VOLUME /models

ENTRYPOINT ["llama-server"]

# --temp/--top-k: the model authors' own guidance for reducing OCR
# hallucination (huggingface.co/blog/ggml-org/using-ocr-models-with-llama-cpp).
CMD ["-hf", "ggml-org/GLM-OCR-GGUF:Q8_0", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "-ngl", "99", \
     "--temp", "0.1", \
     "--top-k", "1"]
