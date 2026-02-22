#!/bin/bash
# MultiAgent-Eval - Tek komutla çalıştır
# Run everything with one command

set -e
cd "$(dirname "$0")"

# Ollama kontrolü (ücretsiz modeller için)
if command -v ollama &> /dev/null; then
    if ! ollama list 2>/dev/null | grep -q mistral; then
        echo "Mistral modeli indiriliyor (ilk seferde birkaç dakika sürebilir)..."
        ollama run mistral --help 2>/dev/null || ollama pull mistral
    fi
else
    echo "Uyarı: Ollama yüklü değil. Ücretsiz modeller için: https://ollama.ai"
    echo "Groq kullanmak için: eval_config.yaml'da primary_model: groq/llama-3.1-70b-versatile"
fi

# PYTHONPATH (pip install yapılmadıysa)
export PYTHONPATH="${PYTHONPATH:-}:${PWD}/src"

echo ""
echo "=== Evaluation çalıştırılıyor ==="
python -m multiagent_eval.cli run --config eval_config.yaml

echo ""
echo "=== Tamamlandı ==="
echo "Sonuçlar: eval_results/result.json"
echo "Rapor: eval_results/report.html"
echo ""
echo "Raporu açmak için: open eval_results/report.html"
