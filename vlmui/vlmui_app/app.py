from flask import Flask, render_template, request, jsonify, send_from_directory, session
from pathlib import Path
from werkzeug.utils import secure_filename
from ..models import ModelIntegrator

curr_dir = Path(__file__).resolve().parent
uploads_dir = curr_dir / "uploads"
uploads_dir.mkdir(exist_ok=True)

vlmui_app = Flask(__name__)
vlmui_app.config["UPLOAD_FOLDER"] = uploads_dir
vlmui_app.config["SECRET_KEY"] = "supersecretkey"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

chat_history = []
model_integrator = ModelIntegrator(device="cpu")

current_model = model_integrator.get_current_model_name()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@vlmui_app.route("/")
def index():
    models = ["CheXagent"]
    return render_template("index.html", models=models, current_model=current_model)


@vlmui_app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400
    filename = secure_filename(file.filename)
    filepath = vlmui_app.config["UPLOAD_FOLDER"] / filename
    model_integrator.set_image_path(filepath)
    file.save(str(filepath))
    return jsonify({"filepath": str(filepath)})


@vlmui_app.route("/set_model", methods=["POST"])
def set_model():
    global current_model
    current_model = request.form.get("model")
    model_integrator.set_model(current_model)
    return jsonify({"current_model": current_model})


@vlmui_app.route("/chat", methods=["POST"])
def chat():
    global chat_history
    if not request.is_json:
        return jsonify({"error": "Invalid request format"}), 400
    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"error": "No message provided"}), 400
    model_response = model_integrator.get_current_model().chat(user_input)
    print(f"{model_integrator._current_model_name}: {model_response}")
    chat_history.append({"sender": "person", "message": user_input})
    chat_history.append({"sender": "model", "message": model_response})
    return jsonify({"response": model_response, "history": chat_history})


@vlmui_app.route("/uploads/<filename>")
def uploaded_file(filename):
    filename = secure_filename(filename)
    return send_from_directory(vlmui_app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    vlmui_app.run(debug=True)
