# Enabling New Op Support in the llama.cpp OpenVINO Backend

## Finding Unsupported Ops

To identify which ops are not yet supported by the OpenVINO backend, follow the instructions in [docs/ops.md](https://github.com/ggml-org/llama.cpp/blob/main/docs/ops.md) to run `test-backend-ops support` and generate a support table.

Once you've identified an unsupported op, check whether it already has an entry in `ggml/src/ggml-openvino/openvino/op_table.cpp`:

- **If the op is NOT in `op_table.cpp`:** The op has no translation yet and you can add support using the instructions below.
- **If the op IS in `op_table.cpp` but shows partial/no support:** This means certain configurations (e.g., specific data types or tensor layouts) of this op are not handled. Extending support for these cases is a more advanced topic.

---

## Beginner Level: Enabling 1-to-1 Op Support

This is the simplest way to enable new ops in the llama.cpp OpenVINO backend. However, this approach requires that the number of inputs/outputs, their layouts, and data types in the GGML operator exactly match those of the corresponding OpenVINO operator.

To use this approach, you only need to update `ggml/src/ggml-openvino/openvino/op_table.cpp`. In this example, we'll enable `GGML_OP_MUL` by mapping it to the OpenVINO `Multiply` operator.

First, insert the include line for the OpenVINO operator you want to add the translation for:

```cpp
#include <openvino/op/multiply.hpp>
```

Then, add a corresponding entry to the `std::unordered_map` returned by `get_supported_ops()` function. There are two helper templates available depending on the number of inputs:

- `op::translate_1to1_match_1_input<OpenVINO_OP>` — for operators with 1 input
- `op::translate_1to1_match_2_inputs<OpenVINO_OP>` — for operators with 2 inputs

For `GGML_OP_MUL` (which takes 2 inputs), the entry should look like:

```cpp
{"GGML_OP_MUL", op::translate_1to1_match_2_inputs<v1::Multiply>}
```

### Example of an Existing OP
A good example of an operator that is already supported 1-to-1 in the llama.cpp OpenVINO backend is `GGML_OP_ADD`. This maps directly to OpenVINO's `Add` operator:
```cpp
#include <openvino/op/add.hpp>
// ...
{"GGML_OP_ADD", op::translate_1to1_match_2_inputs<v1::Add>},
```

### Example of Adding a Missing OP
If you want to add support for Absolute Value (`GGML_OP_ABS`), which processes a single input, you would include the header and map it using the 1-input template:
```cpp
#include <openvino/op/abs.hpp>
// ...
{"GGML_OP_ABS", op::translate_1to1_match_1_input<v0::Abs>},
```

---

## Intermediate Level: Custom Op Translation

Sometimes an operator exists in both GGML and OpenVINO, but a 1-to-1 mapping is impossible. When the GGML operator requires parameter manipulation (e.g., specifying a permutation order), the custom translation must construct those parameters explicitly using the backend's `NodeContext`.

A classic example is `GGML_OP_TRANSPOSE`.

### 1. Implement Custom Translation Logic
Create a new file `ggml/src/ggml-openvino/openvino/op/transpose.cpp`:

```cpp
#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"

#include <openvino/op/transpose.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_transpose(const NodeContext & context) {
    num_inputs_check(context, 1, 1);

    auto res = std::make_shared<ov::op::v1::Transpose>(
        context.get_input(0), ov::op::v0::Constant::create(ov::element::i64, {4}, {0, 1, 3, 2}));
    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
```

Here, unlike the simple 1-to-1 case, the translation constructs a constant permutation vector `{0, 1, 3, 2}` as a second input to the OpenVINO `Transpose` op — this parameter doesn't come from a GGML input but is derived from the operator's semantics.

### 2. Register the Custom Translation
First, declare the translation function in `ggml/src/ggml-openvino/openvino/op_table.h`:

```cpp
GGML_OP_CONVERTER(translate_transpose);
```

Next, add the entry in `get_supported_ops()` in `ggml/src/ggml-openvino/openvino/op_table.cpp`:

```cpp
{"GGML_OP_TRANSPOSE", op::translate_transpose},
```

### 3. Add the Op's Enum Value to the Support Function
If only some configurations of the op are supported, add a case for it in `is_op_unsupported_case` that returns true for any configuration the backend/device doesn't support.

### 4. Add Dynamic Dim Propagation Logic
`GgmlOvDecoder::compute_node_dynamic_dims()` Populates a map that contains information about the dynamic shapes propagated by each node. Extend this function with a new `case` in the `switch(node->op)` block that maps the dynamic dimension from the relevant source tensor to the output node, following the same conventions as existing ops.

For ops where the output shape directly mirrors the input (e.g. `GGML_OP_RMS_NORM`, `GGML_OP_ADD`), simply inherit the dynamic dim index from `src[0]`. For ops that permute, reshape, or stride-reindex dimensions, use stride- or parameter-based matching to determine which output dimension corresponds to the source's dynamic dimension. For ops that produce a statically-shaped output regardless of input, set `m_node_dynamic_dims[node] = -1`.

---

## Next Steps: Verification, Testing, and CI

Whether you are adding a Beginner 1-to-1 mapping or an Intermediate custom translation, you must rigorously test it to ensure it behaves correctly compared to the reference CPU backend.

### 1. Rebuild the Project
Recompile `llama.cpp` to include your changes, ensuring OpenVINO support is enabled:
```bash
cmake -B build -DGGML_OPENVINO=ON
cmake --build build --config Release -j
```

### 2. Verify Backend Ops (Unit Testing)
Run the `test-backend-ops` utility to verify the numerical correctness of your new operator. This test automatically compares the results of the OpenVINO backend against the CPU reference implementation. 

To run tests specifically for your new op (e.g., `GGML_OP_TRANSPOSE`):
```bash
./build/bin/test-backend-ops -b OPENVINO -o TRANSPOSE
```
Ensure all tests for the op pass without any numerical divergence or runtime crashes.

### 3. Check the Support Table
Re-run the support command to visually verify that your op is now correctly registered:
```bash
./build/bin/test-backend-ops support -b OPENVINO
```
Look for your added operator in the output table and verify the configuration matrix reflects successful support.

### 4. End-to-End Model Verification (Optional but Recommended)
If the operator is actively used by a known LLM architecture, run inference on a small model using the OpenVINO backend to ensure no regressions were introduced to the graph execution:
```bash
./build/bin/llama-cli -m model.gguf -p "Hello, world!" -ngl 99
```

### 5. Open a Pull Request and Monitor CI
Once all local tests pass:
1. Commit your changes and push them to your fork.
2. Open a Pull Request against the main `ggml-org/llama.cpp` repository.
3. **Monitor GitHub Actions:** `llama.cpp` has extensive Continuous Integration (CI) pipelines. 
4. Pay special attention to the **OpenVINO CI jobs**. If any of these fail, check the logs—it could indicate that your operator mapping fails on certain OS environments, specific hardware combinations, or edge-case tensor shapes.
5. Address any feedback from the maintainers and CI results by pushing additional commits to your PR branch.
