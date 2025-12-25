import gradio as gr  # type: ignore
from gradio_imageslider import ImageSlider  # type: ignore

import diff_fit.generation_utils.constants as generation_const
from diff_fit.constants import INPAINTING_TABS
from diff_fit.face_generation import FaceGeneration
from diff_fit.face_inpaint import FaceInpaint
from diff_fit.face_transformation import FaceTransformation
from diff_fit.image_slider_fix import _postprocess_image
from diff_fit.transformation_utils.utils import (
    get_points,
    remove_points,
    update_masked_image,
)
from diff_fit.utils import (
    get_inpainting_tab_state,
    get_model_id,
    parse_args,
    reset_index,
    reset_sliders,
    save_file,
    update_history,
    update_selection,
    update_text,
    use_this_output,
)

ImageSlider._postprocess_image = _postprocess_image


def main(
    FaceGeneration,
    FaceTransformation,
    FaceInpaint,
    port=generation_const.SERVER_PORT,
):
    with gr.Blocks() as demo:
        with gr.Tab("Generate images"):
            with gr.Row():
                with gr.Column(scale=1, min_width=300):
                    description = gr.Textbox(label="Description")
                    age = gr.Slider(
                        10, 80, value=30, step=5, label="Age", interactive=True
                    )
                    prompt = [description, age]
                    for (
                        dropdown_name,
                        dropdown_options,
                    ) in generation_const.GENERATION_DROPDOWN.items():
                        dropdown = gr.Dropdown(
                            dropdown_options,
                            label=dropdown_name,
                            interactive=True,
                            allow_custom_value=True,
                            value=dropdown_options[0],
                        )
                        prompt.append(dropdown)

                    with gr.Accordion("Advanced options", open=False):
                        sampling_steps = gr.Slider(
                            1, 50, value=5, step=1, label="Sampling steps"
                        )
                        guidance_scale = gr.Slider(
                            0, 10, value=1, step=0.5, label="Guidance scale"
                        )
                        with gr.Row():
                            seed = gr.Slider(
                                -1,
                                generation_const.MAX_SEED,
                                value=-1,
                                step=1,
                                label="Seed",
                            )
                            batch_count = gr.Slider(
                                1, 10, value=1, step=1, label="Batch count"
                            )
                        with gr.Row():
                            randomize_prompt = gr.Checkbox(
                                label="Randomize prompt", value=True
                            )
                            use_description_only = gr.Checkbox(
                                label="Use description only"
                            )

                with gr.Column(scale=2, min_width=600):
                    generated_images = gr.Gallery(
                        preview=False,
                        allow_preview=True,
                        height=780,
                        interactive=False,
                        type="pil",
                        format="png",
                    )
                    with gr.Row():
                        text2img_button = gr.Button("Generate")
                        use_this_output_image = gr.Button("Use this output")
        with gr.Tab("Img2img"):
            description_img2img = gr.Textbox(label="Description")
            with gr.Accordion("Advanced options", open=False):
                sampling_steps_img2img = gr.Slider(
                    1, 10, value=5, step=1, label="Sampling steps"
                )
                guidance_scale_img2img = gr.Slider(
                    1, 5, value=1, step=0.5, label="Guidance scale"
                )
                strength2 = gr.Slider(0, 1, value=0.8, step=0.1, label="Strength")
                with gr.Row():
                    seed_img2img = gr.Slider(
                        -1, generation_const.MAX_SEED, value=-1, step=1, label="Seed"
                    )
                    batch_count_img2img = gr.Slider(
                        1, 10, value=1, step=1, label="Batch count"
                    )
            with gr.Row():
                with gr.Column():
                    img2img_input = gr.Image(type="pil", height=600, format="png")
                    img2img_button = gr.Button("Generate resembling images")
                with gr.Column():
                    generated_images_img2img = gr.Gallery(
                        preview=True,
                        interactive=False,
                        height=600,
                        type="pil",
                        format="png",
                    )
                    use_this_output_img2img = gr.Button("Use this output")

        with gr.Tab("Inpainting"):
            description_inpaint = gr.Textbox(label="Description")
            with gr.Accordion("Advanced options", open=False):
                sampling_steps_inpaint = gr.Slider(
                    1, 15, value=5, step=1, label="Sampling steps"
                )
                guidance_scale_inpaint = gr.Slider(
                    1, 5, value=1.5, step=0.5, label="Guidance scale"
                )
                strength_inpaint = gr.Slider(
                    0, 1, value=0.8, step=0.1, label="Strength"
                )
                with gr.Row():
                    seed_inpaint = gr.Slider(
                        -1, generation_const.MAX_SEED, value=-1, step=1, label="Seed"
                    )
                    batch_count_inpaint = gr.Slider(
                        1, 10, value=1, step=1, label="Batch count"
                    )
                with gr.Row():
                    crop_inpaint = gr.Checkbox(
                        label="Crop inpaint", value=False, interactive=True
                    )
                    paste_img_back = gr.Checkbox(
                        label="Paste image back", value=True, interactive=True
                    )
            with gr.Tab("Premade masks"):
                inpaint_tab_state = gr.State("Hair")
                inpaint_dropdown_state = gr.State("")

                with gr.Tabs():
                    inpaint_tabs = []
                    for tab_name, dropdown_options in INPAINTING_TABS.items():
                        tab = gr.Tab(tab_name)
                        with tab:
                            dropdown = gr.Dropdown(
                                choices=dropdown_options,
                                allow_custom_value=True,
                                label="Choose an option",
                            )

                            dropdown.change(
                                get_inpainting_tab_state,
                                inputs=dropdown,
                                outputs=inpaint_dropdown_state,
                            )

                            inpaint_tabs.append(tab)

                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Row():
                            inpaint_input_image = gr.Image(
                                type="pil", height=240, interactive=True, format="png"
                            )
                            inpaint_mask = gr.Image(
                                type="pil", height=240, interactive=False, format="png"
                            )
                        inpaint_masked_image = gr.Image(
                            type="pil", height=344, interactive=False, format="png"
                        )
                        inpaint_button2 = gr.Button("Inpaint")
                    with gr.Column(scale=1):
                        # generated_images_inpaint2 = gr.Image(
                        #     height=600, interactive=False
                        # )
                        generated_images_inpaint2 = gr.Gallery(
                            height=600,
                            preview=True,
                            interactive=False,
                            type="pil",
                            format="png",
                        )

                        use_this_output_inpaint2 = gr.Button("Use this output")
            with gr.Tab("Draw mask"):
                selected_image_index = gr.State(0)
                with gr.Row():
                    with gr.Column(scale=1):
                        inpaint_canvas = gr.ImageMask(
                            brush=gr.Brush(colors=["#FFFFFF"], color_mode="fixed"),
                            type="pil",
                            layers=False,
                            height=750,
                            format="png",
                        )
                        inpaint_button = gr.Button("Inpaint")
                    with gr.Column(scale=1):
                        generated_images_inpaint = gr.Gallery(
                            preview=True,
                            height=750,
                            interactive=False,
                            type="pil",
                            format="png",
                        )
                        use_this_output_inpaint = gr.Button("Use this output")

        with gr.Tab("Lightning-drag"):
            with gr.Row():
                with gr.Column(scale=1, min_width=300):
                    with gr.Tab("Premade points"):
                        with gr.Tab("Face shape"):
                            with gr.Tab("Jaw shape"):
                                jawline_vertical_offset = gr.Slider(
                                    -40, 40, value=0, step=5, label="JAWLINE VERTICAL"
                                )
                                jawline_horizontal_offset = gr.Slider(
                                    -40, 40, value=0, step=5, label="JAWLINE HORIZONTAL"
                                )
                            with gr.Tab("Forehead shape"):
                                forehead_vertical_offset = gr.Slider(
                                    -40, 40, value=0, step=5, label="FOREHEAD VERTICAL"
                                )
                                forehead_horizontal_offset = gr.Slider(
                                    -40,
                                    40,
                                    value=0,
                                    step=5,
                                    label="FOREHEAD HORIZONTAL",
                                )
                        with gr.Tab("Eyebrow shape"):
                            brow_vertical_offset = gr.Slider(
                                -40, 40, value=0, step=5, label="BROW VERTICAL"
                            )
                            brow_horizontal_offset = gr.Slider(
                                -40, 40, value=0, step=5, label="BROW HORIZONTAL"
                            )
                        with gr.Tab("Eye shape"):
                            eye_vertical_offset = gr.Slider(
                                -40, 40, value=0, step=5, label="EYE VERTICAL"
                            )
                            eye_horizontal_offset = gr.Slider(
                                -40, 40, value=0, step=5, label="EYE HORIZONTAL"
                            )
                            eye_size_offset = gr.Slider(
                                -10, 10, value=0, step=1, label="EYE SIZE"
                            )
                            eye_rotation_offset = gr.Slider(
                                -45, 45, value=0, step=5, label="EYE ROTATION"
                            )
                        with gr.Tab("Nose shape"):
                            nose_vertical_offset = gr.Slider(
                                -40, 40, value=0, step=5, label="NOSE VERTICAL"
                            )
                            nose_horizontal_offset = gr.Slider(
                                -40, 40, value=0, step=5, label="NOSE HORIZONTAL"
                            )
                            nose_width_offset = gr.Slider(
                                -20, 20, value=0, step=1, label="NOSE WIDTH"
                            )
                        with gr.Tab("Lips shape"):
                            lips_vertical_offset = gr.Slider(
                                -40, 40, value=0, step=5, label="LIPS VERTICAL"
                            )
                            lips_horizontal_offset = gr.Slider(
                                -40, 40, value=0, step=5, label="LIPS HORIZONTAL"
                            )
                            lips_volume_offset = gr.Slider(
                                -10, 10, value=0, step=1, label="LIPS VOLUME"
                            )
                        with gr.Row():
                            image_transformation = gr.Image(
                                type="pil", interactive=True, format="png"
                            )
                    with gr.Tab("Custom points"):
                        selected_points = gr.State([])
                        with gr.Tab("Draw mask"):
                            transformation_canvas = gr.ImageMask(
                                brush=gr.Brush(colors=["#FFFFFF"], color_mode="fixed"),
                                type="pil",
                                layers=False,
                                height=550,
                            )
                        with gr.Tab("Select points"):
                            masked_image = gr.Image(
                                type="pil", interactive=False, format="png"
                            )
                        with gr.Row():
                            remove_points_btn = gr.Button("Remove points")
                            run_btn = gr.Button("Run transformation")
                with gr.Column(scale=1.5):
                    with gr.Row():
                        image_transformation_slider = ImageSlider(
                            type="pil", interactive=False, height=800
                        )
                    with gr.Row():
                        continue_tranforming = gr.Button("Continue with transformation")
                        use_this_output_transform = gr.Button("Use this output")

        history_gallery = gr.Gallery(
            label="History",
            columns=12,
            object_fit="contain",
            interactive=False,
            height=130,
            type="pil",
            format="png",
        )

        target_name = gr.Textbox(label="Save Directory")

        scheduler_dropdown = gr.Dropdown(
            [
                "sde",
                "sde_karras",
                "sde_my_way",
                "sde_karras_my_way",
                "single",
                "single_karras",
                "multi",
                "multi_karras",
                "TCD",
                "Euler",
                "random",
            ],
            label="Scheduler",
            value="single",
        )

        OFFSETS = [
            jawline_vertical_offset,
            jawline_horizontal_offset,
            forehead_vertical_offset,
            forehead_horizontal_offset,
            brow_vertical_offset,
            brow_horizontal_offset,
            eye_vertical_offset,
            eye_horizontal_offset,
            eye_size_offset,
            eye_rotation_offset,
            nose_vertical_offset,
            nose_horizontal_offset,
            nose_width_offset,
            lips_vertical_offset,
            lips_horizontal_offset,
            lips_volume_offset,
        ]

        INPUTS = [
            img2img_input,
            inpaint_canvas,
            inpaint_input_image,
        ]
        OUTPUTS = [
            generated_images,
            generated_images_img2img,
            generated_images_inpaint,
            generated_images_inpaint2,
        ]
        USE_THIS_OUTPUTS = [
            use_this_output_image,
            use_this_output_img2img,
            use_this_output_inpaint,
            use_this_output_inpaint2,
        ]

        temp_pipeline = gr.Textbox(
            visible=False,
        )
        temp_scheduler = gr.Textbox(
            visible=False,
        )
        temp_karras = gr.Textbox(
            visible=False,
        )
        temp_seed = gr.Textbox(
            visible=False,
        )
        temp_prompt = gr.Textbox(
            visible=False,
        )
        temp_negative_prompt = gr.Textbox(
            visible=False,
        )
        temp_steps = gr.Textbox(
            visible=False,
        )
        temp_cfg = gr.Textbox(
            visible=False,
        )
        temp_strength = gr.Textbox(
            visible=False,
        )
        temp_points = gr.Textbox(
            visible=False,
        )
        temp_mask = gr.Image(type="pil", visible=False, format="png")
        temp_transformation = gr.Image(type="pil", visible=False, format="png")
        temp_output_image = gr.Image(type="pil", visible=False, format="png")

        TEMPS = [
            temp_pipeline,
            temp_scheduler,
            temp_karras,
            temp_seed,
            temp_prompt,
            temp_negative_prompt,
            temp_steps,
            temp_cfg,
            temp_strength,
            temp_points,
            temp_mask,
            temp_transformation,
            temp_output_image,
        ]

        text2img_button.click(
            FaceGeneration.txt2image,
            inputs=[
                sampling_steps,
                guidance_scale,
                batch_count,
                randomize_prompt,
                seed,
                scheduler_dropdown,
                use_description_only,
            ]
            + prompt,
            outputs=[generated_images] + TEMPS,
        )
        img2img_button.click(
            FaceGeneration.img2img,
            inputs=[
                img2img_input,
                description_img2img,
                sampling_steps_img2img,
                seed_img2img,
                guidance_scale_img2img,
                strength2,
                batch_count_img2img,
                scheduler_dropdown,
            ],
            outputs=[generated_images_img2img] + TEMPS,
        )
        inpaint_button.click(
            FaceGeneration.inpaint,
            inputs=[
                inpaint_canvas,
                description_inpaint,
                sampling_steps_inpaint,
                seed_inpaint,
                guidance_scale_inpaint,
                strength_inpaint,
                batch_count_inpaint,
                scheduler_dropdown,
                crop_inpaint,
                paste_img_back,
            ],
            outputs=[generated_images_inpaint] + TEMPS,
        )

        inpaint_button2.click(
            FaceGeneration.inpaint,
            inputs=[
                inpaint_input_image,
                description_inpaint,
                sampling_steps_inpaint,
                seed_inpaint,
                guidance_scale_inpaint,
                strength_inpaint,
                batch_count_inpaint,
                scheduler_dropdown,
                crop_inpaint,
                paste_img_back,
                inpaint_mask,
                inpaint_dropdown_state,
                inpaint_tab_state,
            ],
            outputs=[generated_images_inpaint2] + TEMPS,
        )

        inpaint_input_image.change(
            FaceInpaint.init_inpaint,
            inputs=[inpaint_input_image],
        ).then(
            FaceInpaint.show_mask,
            inputs=[
                inpaint_input_image,
                inpaint_tab_state,
            ],
            outputs=[inpaint_mask, inpaint_masked_image],
        )

        for tab in inpaint_tabs:
            tab.select(
                get_inpainting_tab_state,
                inputs=gr.State(value=tab.label),
                outputs=inpaint_tab_state,
            ).then(
                FaceInpaint.show_mask,
                inputs=[
                    inpaint_input_image,
                    inpaint_tab_state,
                ],
                outputs=[inpaint_mask, inpaint_masked_image],
            )

        transformation_canvas.upload(
            fn=lambda x: FaceTransformation.init_image(
                x["background"], detect_landmarks=False
            ),
            inputs=[transformation_canvas],
            outputs=[image_transformation, transformation_canvas],
        )

        transformation_canvas.change(
            fn=update_masked_image,
            inputs=[transformation_canvas],
            outputs=[masked_image],
        )

        masked_image.select(
            get_points,
            [masked_image, selected_points],
            [masked_image],
        )

        run_btn.click(
            fn=lambda points, x: FaceTransformation.execute_transform(
                points, x["layers"][0]
            ),
            inputs=[selected_points, transformation_canvas],
            outputs=[
                image_transformation_slider,
                gr.Image(type="pil", visible=False, format="png"),
            ]
            + TEMPS,
        )

        remove_points_btn.click(
            remove_points,
            inputs=[],
            outputs=selected_points,
        ).then(
            fn=update_masked_image,
            inputs=[transformation_canvas],
            outputs=[masked_image],
        )

        image_transformation.upload(
            reset_sliders,
            inputs=OFFSETS,
            outputs=OFFSETS,
        ).then(
            FaceTransformation.init_image,
            inputs=[image_transformation],
            outputs=[],
        )

        for offset in OFFSETS:
            offset.release(
                FaceTransformation.transform,
                inputs=OFFSETS,
                outputs=[image_transformation_slider, image_transformation] + TEMPS,
            )

        continue_tranforming.click(
            reset_sliders,
            inputs=OFFSETS,
            outputs=OFFSETS,
        ).then(
            FaceTransformation.change_output,
            inputs=[image_transformation_slider],
            outputs=[image_transformation, transformation_canvas],
        ).then(
            fn=lambda x, history: update_history(x[1], history),
            inputs=[image_transformation_slider, history_gallery],
            outputs=history_gallery,
        ).then(
            remove_points,
            inputs=[],
            outputs=selected_points,
        ).then(
            save_file, inputs=[target_name] + TEMPS + [crop_inpaint]
        )

        use_this_output_transform.click(
            use_this_output,
            inputs=[
                image_transformation_slider,
                gr.State(len(INPUTS)),
                gr.State(-1),
            ],
            outputs=INPUTS,
        ).then(
            reset_sliders,
            inputs=OFFSETS,
            outputs=OFFSETS,
        ).then(
            fn=lambda x: FaceTransformation.init_image(x[1]),
            inputs=[image_transformation_slider],
            outputs=[image_transformation, transformation_canvas],
        ).then(
            fn=lambda x, history: update_history(x[1], history),
            inputs=[image_transformation_slider, history_gallery],
            outputs=history_gallery,
        ).then(
            remove_points,
            inputs=[],
            outputs=selected_points,
        ).then(
            save_file, inputs=[target_name] + TEMPS + [crop_inpaint]
        )

        for output in OUTPUTS:
            output.select(
                update_selection,
                inputs=None,
                outputs=[selected_image_index],
            )

        for use_this_output_btn, output in zip(USE_THIS_OUTPUTS, OUTPUTS):
            use_this_output_btn.click(
                use_this_output,
                inputs=[
                    output,
                    gr.State(len(INPUTS)),
                    selected_image_index,
                ],
                outputs=INPUTS,
            ).then(
                reset_sliders,
                inputs=OFFSETS,
                outputs=OFFSETS,
            ).then(
                FaceTransformation.init_image,
                inputs=[output, selected_image_index],
                outputs=[image_transformation, transformation_canvas],
            ).then(
                fn=update_history,
                inputs=[output, history_gallery, selected_image_index],
                outputs=history_gallery,
            ).then(
                reset_index,
                inputs=[],
                outputs=[selected_image_index],
            ).then(
                remove_points,
                inputs=[],
                outputs=selected_points,
            ).then(
                save_file, inputs=[target_name] + TEMPS + [crop_inpaint]
            )

            use_this_output_image.click(
                update_text, inputs=temp_prompt, outputs=description_img2img
            )

    demo.launch(server_port=port)


def run_diff_fit():
    args = parse_args()
    model_id = get_model_id(args.model)
    print(f"Using model: {model_id}")
    main(
        FaceGeneration=FaceGeneration(model_id),
        FaceTransformation=FaceTransformation(),
        FaceInpaint=FaceInpaint(),
        port=args.port,
    )


if __name__ == "__main__":
    run_diff_fit()
