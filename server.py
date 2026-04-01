"""
========================================================================
 🩺 AB-GEN MEDICAL DEMO — Servidor Flask
 Dashboard de diagnóstico clínico con trazabilidad XAI
========================================================================
"""

import os
import sys
import numpy as np
from flask import Flask, render_template, jsonify, request

# Asegurar que el directorio padre está en el path para cargar modelos
DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(DEMO_DIR)

app = Flask(__name__)

# Motor global
engine = None


def get_engine():
    global engine
    if engine is None:
        from engine import MedicalEngine
        engine = MedicalEngine(models_dir=PARENT_DIR)
    return engine


@app.route("/")
def index():
    return render_template("index.html")


import json
from engine import sanitize_for_json

@app.route("/api/analyze", methods=["POST"])
def analyze():
    import json
    try:
        eng = get_engine()
        data = request.get_json(silent=True) or {}
        index = data.get("index", None)
        result = eng.analyze_sample(index=index)
        
        # ULTIMATE NATIVE CONVERTER
        def to_native(o):
            if isinstance(o, dict): return {str(k): to_native(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)): return [to_native(v) for v in o]
            if hasattr(o, "tolist"): return to_native(o.tolist())
            if hasattr(o, "item"): return o.item()
            return o

        return app.response_class(
            json.dumps(to_native(result), default=str),
            mimetype='application/json'
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"API_ERROR: {str(e)}"}), 500


@app.route("/api/batch", methods=["POST"])
def batch():
    import json
    try:
        eng = get_engine()
        data = request.get_json(silent=True) or {}
        n = data.get("n", 10)
        result = eng.analyze_batch(n=min(int(n), 50))
        
        def to_native(o):
            if isinstance(o, dict): return {str(k): to_native(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)): return [to_native(v) for v in o]
            if hasattr(o, "tolist"): return to_native(o.tolist())
            if hasattr(o, "item"): return o.item()
            return o

        return app.response_class(
            json.dumps(to_native(result), default=str),
            mimetype='application/json'
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/stats", methods=["GET"])
def stats():
    try:
        eng = get_engine()
        info = eng.get_model_info()
        return jsonify(info)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("\n[+] AB-GEN MEDICAL DEMO - Iniciando servidor...")
    print("    Cargando modelos (primera vez puede tardar ~30s)...\n")

    # Pre-cargar motor
    get_engine()

    print("\n" + "=" * 60)
    print("  [OK] Dashboard disponible en: http://localhost:5000")
    print("=" * 60 + "\n")

    app.run(host="127.0.0.1", port=5000, debug=True)
