# SketchRec Mini ✍🏻 App Python

Sketch recognition is a subfield of artificial intelligence and computer vision that focuses on recognizing sketches and hand-drawn images. It involves the development of algorithms and techniques that can interpret hand-drawn images and identify the objects or concepts they represent.

Sketch recognition has numerous applications, including in education, design, and engineering. For example, it can be used to recognize hand-drawn diagrams and schematics, which can help students and professionals communicate and collaborate more effectively.

## Requirements

To run this example, you need to have Python 3.x and the following libraries installed:

- Gradio
- Pandas
- PyTorch

You can install these libraries using pip:

```
pip install gradio pandas torch
```

## Usage

To run the Gradio app, simply run the `app.py` file:

```
python app.py
```

This will start the Gradio app and open it in your default web browser. You can then enter a text in the input box, click on "Submit", and the app will output the sentiment prediction.

## Customization

You can customize the Gradio interface by modifying the `gradio.Interface` object in the `app.py` file. For example, you can change the input and output types, add new input fields, or modify the styling.

## Acknowledgments

This example is based on the [Gradio Quickstart](https://gradio.app/docs/quickstart) and the [Sentiment Analysis Example](https://github.com/gradio-app/huggingface-transformers-demo) by the Gradio team.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
title: Sketch Rec Mini
emoji: 🏢
colorFrom: gray
colorTo: green
sdk: gradio
sdk_version: 3.29.0
app_file: app.py
pinned: false
license: mit
---