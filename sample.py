import gradio as gr
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")  # ✅ use non-GUI backend
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

def detect_defects_with_yolo(image):
    if image is None:
        return None, [], "No image provided", "N/A"

    try:
        print("[DEBUG] Converting image")
        result_image = image.copy()
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb)

        print("[DEBUG] Running YOLO prediction")
        results = model.predict(pil_img, verbose=False)[0]
        print("[DEBUG] Prediction complete")

        defect_classes = []
        boxes = results.boxes
        names = model.names

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = names[cls] if cls in names else f"class_{cls}"
            defect_classes.append(label)
            color = (0, 0, 255)
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            text_size = cv2.getTextSize(f"{label} {conf:.2f}", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(result_image, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), color, -1)
            cv2.putText(result_image, f"{label} {conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        status = "Pass" if not defect_classes or all(cls.lower() == "no defect" for cls in defect_classes) else "Defective"
        return np.array(result_image), defect_classes, status

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, [], f"Detection Error: {str(e)}", "N/A"

def create_temperature_chart():
    try:
        fig, ax = plt.subplots(figsize=(5, 3))
        timestamp = np.linspace(0, 60, 100)
        base_temp = 25 + 2 * np.sin(timestamp / 10)
        noise = np.random.normal(0, 0.3, 100)
        trend = np.linspace(0, 1.5, 100)
        temperature = base_temp + noise + trend
        ax.plot(timestamp, temperature, color='#1e88e5', linewidth=2)
        ax.set_title("Temperature Over Time")
        ax.set_ylabel("Temperature (°C)")
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 60])
        ax.set_ylim([20, 35])
        ax.set_xticks([0, 15, 30, 45, 60])
        ax.set_xticklabels(['60s', '45s', '30s', '15s', 'Now'])
        plt.tight_layout()
        return fig
    except Exception as e:
        print("Temperature chart error:", e)
        return None

def create_continuity_chart():
    try:
        fig, ax = plt.subplots(figsize=(5, 3))
        points = np.linspace(0, 10, 100)
        base = 5 + 2 * np.sin(points) + np.sin(2 * points)
        noise = np.random.normal(0, 0.2, 100)
        continuity = base + noise
        ax.plot(points, continuity, color='#1e88e5', linewidth=2)
        ax.set_title("Continuity Results")
        ax.set_ylabel("Resistance (Ω)")
        ax.grid(True, alpha=0.3)
        ax.set_xticks([])
        plt.tight_layout()
        return fig
    except Exception as e:
        print("Continuity chart error:", e)
        return None

def process_pcb_image(input_image):
    print("[DEBUG] process_pcb_image called")
    if input_image is None:
        print("[DEBUG] No input image")
        return None, None, "No image provided", "N/A", None

    result_image, defects, status = detect_defects_with_yolo(input_image)
    defect_list = ", ".join(defects) if defects else "None detected"
    status_color = "red" if status == "Defective" else "green"
    status_html = f"<span style='color: {status_color}; font-weight: bold; font-size: 18px;'>{status}</span>"
    temp_chart = create_temperature_chart()
    cont_chart = create_continuity_chart()
    print("[DEBUG] Detection done, returning values")
    return result_image, defect_list, status_html, temp_chart, cont_chart

custom_css = """
#sidebar {
background-color: #1e293b;
color: white;
border-radius: 10px;
margin-right: 10px;
height: 100%;
}
.sidebar-item {
padding: 10px;
margin: 5px 0;
border-radius: 5px;
cursor: pointer;
transition: background-color 0.3s;
}
.sidebar-item:hover {
background-color: #2c3e50;
}
.sidebar-item.active {
background-color: #3498db;
}
.sidebar-icon {
margin-right: 10px;
}
.header-text {
font-size: 24px;
font-weight: bold;
margin-bottom: 20px;
color: white;
}
.card {
border-radius: 10px;
box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
padding: 15px;
margin-bottom: 15px;
background-color: white;
}
.card-header {
font-size: 18px;
font-weight: bold;
margin-bottom: 10px;
color: #1e293b;
}
"""

def create_pcb_detection_ui():
    theme = gr.themes.Base(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
    ).set(
        body_background_fill="#f8fafc",
        block_background_fill="#ffffff",
        block_label_background_fill="#1e293b",
        block_label_text_color="#ffffff",
        button_primary_background_fill="#1e88e5",
    )

    with gr.Blocks(theme=theme, css=custom_css) as app:
        gr.Markdown("# PCB Defect Detection")

        with gr.Row():
            with gr.Column(scale=1, elem_id="sidebar"):
                gr.Markdown('<div class="header-text">PCB Defect Detection</div>')
                with gr.Group():
                    run_temp = gr.Button("🌡️ Temperature Check", variant="secondary")
                    run_cont = gr.Button("⚡ Continuity Test", variant="secondary")

            with gr.Column(scale=4):
                input_image = gr.Image(sources=["upload", "webcam"], type="numpy", label="Upload PCB Image")
                output_image = gr.Image(type="numpy", label="Annotated Output")
                defect_text = gr.Textbox(label="Defect:")
                status_html = gr.HTML(label="Status:")
                temp_chart = gr.Plot(label="Temperature Chart")
                cont_chart = gr.Plot(label="Continuity Chart")

                input_image.change(
                    fn=process_pcb_image,
                    inputs=input_image,
                    outputs=[output_image, defect_text, status_html, temp_chart, cont_chart],
                )

                run_temp.click(fn=create_temperature_chart, outputs=temp_chart)
                run_cont.click(fn=create_continuity_chart, outputs=cont_chart)

    return app

if __name__ == "__main__":
    app = create_pcb_detection_ui()
    app.launch()
