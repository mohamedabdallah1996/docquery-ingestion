FROM ghcr.io/ggml-org/llama.cpp:server-cuda

# LLAMA_CACHE is the documented cache dir for `-hf` downloads
ENV LLAMA_CACHE=/models
ENV HF_HOME=/models

VOLUME /models

ENTRYPOINT ["/app/llama-server"]

CMD ["-hf", "ggml-org/GLM-OCR-GGUF:Q8_0", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "-ngl", "99", \
     "--temp", "0.1", \
     "--top-k", "1"]
