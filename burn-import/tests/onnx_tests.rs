#[cfg(test)]
#[cfg(feature = "onnx")]
mod tests {
    use std::fs::read_to_string;
    use std::path::Path;

    use burn::record::FullPrecisionSettings;
    use pretty_assertions::assert_eq;
    use rstest::*;

    fn code<P: AsRef<Path>>(onnx_path: P) -> String {
        let graph = burn_import::onnx::parse_onnx(onnx_path.as_ref());
        let graph = graph
            .into_burn::<FullPrecisionSettings>()
            .with_blank_space(true)
            .with_top_comment(Some("Generated by integration tests".into()));

        burn_import::format_tokens(graph.codegen())
    }

    #[rstest]
    #[case::mixed("model1")]
    #[case::conv2d("conv2d")]
    // #[case::description_here("model2")] <- Add more models here
    fn test_codegen(#[case] model_name: &str) {
        let input_file = format!("tests/data/{model_name}/{model_name}.onnx");
        let source_file = format!("tests/data/{model_name}/{model_name}.rs");
        let source_expected: String =
            read_to_string(source_file).expect("Expected source file is missing");

        let generated_code = code(input_file);

        // Uncomment this to update the expected code
        // println!("Generated code:\n{}", generated_code);

        assert_eq!(
            source_expected, generated_code,
            "Expected code is left, actual code is right"
        );
    }
}
