# Enabling New Op Support in the llama.cpp OpenVINO Backend

This guide walks you through adding support for a new GGML operator in the
OpenVINO backend, from the simplest 1-to-1 mapping through to ops that need
custom parameter handling, variant dispatch, and device-specific gating.

**Files you will touch:**

| File | Purpose |
|---|---|
| `ggml/src/ggml-openvino/openvino/op_table.cpp` | Registers op name -> translation function |
| `ggml/src/ggml-openvino/openvino/op_table.h` | Declares custom translation functions |
| `ggml/src/ggml-openvino/openvino/op/<op>.cpp` | Your custom translation logic |
| `ggml/src/ggml-openvino/ggml-openvino.cpp` | `is_op_unsupported_case()` — rejects configurations |
| `ggml/src/ggml-openvino/ggml-decoder.cpp` | `compute_op_case()`, `compute_node_dynamic_dims()` |

---

## Finding Unsupported Ops

To identify which ops are not yet supported, follow the instructions in
[docs/ops.md](https://github.com/ggml-org/llama.cpp/blob/main/docs/ops.md) and
run the support command:

```bash
./build/bin/test-backend-ops support -b OPENVINO
```

This prints a matrix of every op and the configurations the backend supports.

### Know Your Op's Registry Key

Before searching `op_table.cpp`, work out which of the **three** key prefixes
your op uses. This trips up almost every new contributor: `test-backend-ops`
prints the *short* name (`RELU`, `GEGLU_QUICK`), but the registry key is
prefixed, and the prefix depends on the op kind.

| Kind | How GGML stores it | Registry key format | Example |
|---|---|---|---|
| Regular op | `ggml_tensor.op` | `GGML_OP_<NAME>` | `GGML_OP_POOL_2D` |
| Unary op | `op == GGML_OP_UNARY`, kind in `op_params[0]` | `GGML_UNARY_OP_<NAME>` | `GGML_UNARY_OP_RELU` |
| GLU op | `op == GGML_OP_GLU`, kind in `op_params[0]` | `GGML_GLU_OP_<NAME>` | `GGML_GLU_OP_GEGLU_QUICK` |

If `test-backend-ops` reports `RELU` as unsupported, the key you add is
`GGML_UNARY_OP_RELU` — grepping for `GGML_OP_RELU` returns nothing even
though the op is perfectly real.

The enum lists live in `ggml/include/ggml.h` (`ggml_op`, `ggml_unary_op`,
`ggml_glu_op`). Check there if unsure which bucket your op falls into.

### Triage: What Kind of Work Is Needed?

Once you know the correct key, search for it in `op_table.cpp`:

1. **The key is absent entirely.** The op has no translation yet. Start at the
   Beginner section below.

2. **You searched for the wrong prefix.** Re-check the table above before
   concluding the op is missing. This is by far the most common false alarm.

3. **The key is present, but the support matrix shows partial/no support.**
   The translation exists but certain configurations (data types, layouts,
   devices, or op modes) are rejected or mistranslated. Three places decide this:
   - `is_op_unsupported_case()` in `ggml-openvino.cpp` — returns `true` to
     *reject* a configuration. If your case is refused outright, the gate is
     probably here.
   - `GgmlOvDecoder::compute_op_case()` in `ggml-decoder.cpp` — classifies op
     variants. An `op_case` of `0` means "unrecognised variant".
   - The translation function itself in `openvino/op/<op>.cpp` — may only
     handle a subset of parameter combinations.

---

## Read This First: GGML Dimension Order Is Reversed

GGML stores dimensions **fastest-varying first**: `ne[0]` is the innermost
(contiguous) dimension. OpenVINO uses conventional slowest-first, NCHW-style
ordering. **Whenever you copy shapes, strides, kernel sizes, padding, axes, or
per-axis parameters from GGML into an OpenVINO op, you must reverse them.**

```cpp
// GGML op_params for POOL_2D: [mode, k0, k1, s0, s1, p0, p1]
// k0/s0/p0 apply to ne[0] (innermost) -> becomes the LAST OpenVINO dim
ov::Shape   kernel    {(size_t) k1, (size_t) k0};   // k1 first
ov::Strides strides   {(size_t) s1, (size_t) s0};
ov::Shape   pads_begin{(size_t) p1, (size_t) p0};
ov::Shape   pads_end  {(size_t) p1, (size_t) p0};
```

```cpp
// GGML ROLL shifts s0..s3 map onto OpenVINO axes 0..3 in reverse
auto shift = ov::op::v0::Constant::create(
    ov::element::i64, ov::Shape{4}, std::vector<int64_t>{s3, s2, s1, s0});
auto axes  = ov::op::v0::Constant::create(
    ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 1, 2, 3});
```

Getting this wrong compiles cleanly and produces plausible-looking but
incorrect numbers. **Always verify on a non-square, non-symmetric shape** — a
test with equal kernel dimensions passes even when the order is inverted.

---

## Beginner Level: Enabling 1-to-1 Op Support

This is the simplest way to enable a new op. It requires that the number of
inputs/outputs, their layouts, and their data types in the GGML operator
exactly match the OpenVINO operator, with no parameter translation needed.

You only need to touch `ggml/src/ggml-openvino/openvino/op_table.cpp`. In this
example we enable `GGML_OP_MUL` by mapping it to OpenVINO's `Multiply`.

First, add the include for the OpenVINO operator:

```cpp
#include <openvino/op/multiply.hpp>
```

Then add an entry to the `std::unordered_map` returned by
`get_supported_ops()`. Two helper templates exist depending on input count:

- `op::translate_1to1_match_1_input<OpenVINO_OP>` — for operators with 1 input
- `op::translate_1to1_match_2_inputs<OpenVINO_OP>` — for operators with 2 inputs

For `GGML_OP_MUL` (2 inputs):

```cpp
{"GGML_OP_MUL", op::translate_1to1_match_2_inputs<v1::Multiply>},
```

### Example of an Existing Op

`GGML_OP_ADD` is already supported this way:

```cpp
#include <openvino/op/add.hpp>
// ...
{"GGML_OP_ADD", op::translate_1to1_match_2_inputs<v1::Add>},
```

`GGML_UNARY_OP_RELU` is the unary equivalent:

```cpp
#include <openvino/op/relu.hpp>
// ...
{"GGML_UNARY_OP_RELU", op::translate_1to1_match_1_input<v0::Relu>},
```

### Example of Adding a Missing Op

To add Absolute Value, include the header and map it with the 1-input template:

```cpp
#include <openvino/op/abs.hpp>
// ...
{"GGML_UNARY_OP_ABS", op::translate_1to1_match_1_input<v0::Abs>},
```

> **Note the prefix.** `abs` is a *unary* op in GGML — it arrives as
> `GGML_OP_UNARY` with `GGML_UNARY_OP_ABS` in `op_params[0]`. A key of
> `"GGML_OP_ABS"` would never be looked up and the op would silently remain
> unsupported.

### When 1-to-1 Is *Not* Enough

Move to the Intermediate section if any of these are true:

- The op carries configuration in `op_params` (kernel size, axes, shifts, flags).
- The OpenVINO op needs extra inputs GGML doesn't provide (permutation vectors,
  axis constants).
- One GGML op maps to several OpenVINO ops depending on a mode flag.
- Inputs may be views (non-contiguous or offset tensors).
- Input arity varies between invocations.

---

## Intermediate Level: Custom Op Translation

Every custom translation follows the same skeleton:

```cpp
#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"

// ... OpenVINO op headers ...

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_<name>(const NodeContext & context) {
    num_inputs_check(context, <min>, <max>);
    // ... build OpenVINO nodes ...
    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
```

The three examples below build up in difficulty. Read them in order.

### Example A (Warm-up): `GGML_OP_TRANSPOSE` — Synthesising a Constant Input

OpenVINO's `Transpose` needs a permutation vector as a second input, but GGML
provides no such input. Create
`ggml/src/ggml-openvino/openvino/op/transpose.cpp`:

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
        context.get_input(0),
        ov::op::v0::Constant::create(ov::element::i64, {4}, {0, 1, 3, 2}));
    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
```

The permutation `{0, 1, 3, 2}` is built as a constant rather than read from a
GGML input — that is the whole difference from the 1-to-1 case.

### Example B: `GGML_OP_ROLL` — Reading `op_params`

Most non-trivial ops carry configuration in `ggml_tensor.op_params`, retrieved
with `context.get_output_op_params()`:

```cpp
OutputVector translate_roll(const NodeContext & context) {
    num_inputs_check(context, 1, 1);
    const int32_t * params = context.get_output_op_params();

    int64_t s0 = params[0];
    int64_t s1 = params[1];
    int64_t s2 = params[2];
    int64_t s3 = params[3];

    auto input = context.get_input(0);

    // NOTE the reversal: s3 first (see "Dimension Order Is Reversed")
    auto shift = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{4}, std::vector<int64_t>{s3, s2, s1, s0});
    auto axes  = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 1, 2, 3});

    auto roll = std::make_shared<ov::op::v7::Roll>(input, shift, axes);
    return rename_outputs_with_suffix({roll}, context.get_name());
}
```

**Finding the `op_params` layout:** indices are op-specific and undocumented.
The ground truth is how the CPU backend unpacks them — read the matching
`ggml_compute_forward_<op>` in `ggml/src/ggml-cpu/ops.cpp`.

### Example C: `GGML_OP_POOL_2D` — Variant Dispatch with `op_case`

When one GGML op maps to several OpenVINO ops, classify the variant in
`GgmlOvDecoder::compute_op_case()` in `ggml-decoder.cpp`:

```cpp
case GGML_OP_POOL_2D: {
    const ggml_op_pool pool_mode = static_cast<ggml_op_pool>(node->op_params[0]);
    switch (pool_mode) {
    case GGML_OP_POOL_MAX: op_case = 1; break;
    case GGML_OP_POOL_AVG: op_case = 2; break;
    default:               op_case = 0; break;   // 0 == unrecognised
    }
    break;
}
```

Then dispatch on it inside the translation via `context.get_op_case()`:

```cpp
OutputVector translate_pool_2d(const NodeContext & context) {
    num_inputs_check(context, 1, 1);
    const int32_t * params = context.get_output_op_params();

    const int k0 = params[1], k1 = params[2];
    const int s0 = params[3], s1 = params[4];
    const int p0 = params[5], p1 = params[6];

    ov::Output<Node> input = context.get_input(0);
    ov::Strides strides{(size_t) s1, (size_t) s0};
    ov::Shape pads_begin{(size_t) p1, (size_t) p0};
    ov::Shape pads_end  {(size_t) p1, (size_t) p0};
    ov::Shape kernel    {(size_t) k1, (size_t) k0};
    ov::Output<Node> res;

    switch (context.get_op_case()) {
    case 1:  // GGML_OP_POOL_MAX
        res = std::make_shared<ov::op::v1::MaxPool>(
            input, strides, pads_begin, pads_end, kernel);
        break;
    case 2:  // GGML_OP_POOL_AVG
        res = std::make_shared<ov::op::v1::AvgPool>(
            input, strides, pads_begin, pads_end, kernel, false);
        break;
    default:
        break;
    }
    return rename_outputs_with_suffix({res}, context.get_name());
}
```

### Two Traps: Views and Data Types

**Views.** Never call `context.get_input(i)` directly when the input may be a
view (non-contiguous or offset tensor). Use:

```cpp
auto src0 = process_view_input_new(context, 0);
```

This inserts the slicing/reshaping needed so the OpenVINO node sees the same
data the CPU backend would. Using `get_input` on a view compiles and runs but
reads the wrong memory.

**Data types.** OpenVINO is strict: an f16 tensor multiplied by an f32 constant
throws at graph-build time. Always derive constant type from the input:

```cpp
auto input_type = src0.get_element_type();
auto coef = ov::op::v0::Constant::create(input_type, ov::Shape{}, {1.702f});
```

**Variable arity.** Some ops (e.g. GLU family) accept either one packed tensor
or two separate ones. Declare the range and branch:

```cpp
num_inputs_check(context, 1, 2);
if (context.get_input_size() == 2) { /* two tensors */ }
else                               { /* split one tensor along last axis */ }
```

### Registering the Custom Translation

Declare the function in `ggml/src/ggml-openvino/openvino/op_table.h`:

```cpp
GGML_OP_CONVERTER(translate_pool_2d);
```

Then add the entry in `get_supported_ops()` in `op_table.cpp`:

```cpp
{"GGML_OP_POOL_2D", op::translate_pool_2d},
```

### Gating Unsupported Configurations

If only some configurations work, add a case in `is_op_unsupported_case()` in
`ggml-openvino.cpp` returning `true` for anything the backend can't handle.
Limitations are often **device-specific** — gate narrowly rather than
disabling the op everywhere:

```cpp
case GGML_OP_POOL_2D: {
    const auto & name = ggml_openvino_get_device_name();
    if (name == "GPU" || name == "NPU") {
        const int32_t * params = op->op_params;
        const int k0 = params[1], k1 = params[2];
        const int p0 = params[5], p1 = params[6];
        if ((p0 > 0 || p1 > 0) && (k0 < 3 || k1 < 3)) {
            return true;   // true == unsupported
        }
    }
    break;
}
```

Keep these conditions as tight as the real hardware limitation. Over-broad
gates silently push work back to the CPU backend and cost performance.

### Adding Dynamic Dim Propagation Logic

`GgmlOvDecoder::compute_node_dynamic_dims()` populates a map describing dynamic
shapes propagated by each node. Add a `case` in its `switch (node->op)`.

For ops whose output shape mirrors the input (e.g. `GGML_OP_RMS_NORM`,
`GGML_OP_ADD`), inherit the dynamic dim index from `src[0]`. For ops that
permute, reshape, or stride-reindex, map the input's dynamic axis onto its new
position in the output — the same reversal caveat applies.

---

## Next Steps: Verification, Testing, and CI

Whether you added a Beginner 1-to-1 mapping or an Intermediate custom translation, test it rigorously against the reference CPU backend before opening a PR.

### 1. Rebuild the Project

Use the same build layout as the rest of this guide so paths match:

```bash
# Linux
source /opt/intel/openvino/setupvars.sh
cmake -B build/ReleaseOV -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_OPENVINO=ON
cmake --build build/ReleaseOV --parallel
```

See [contributing-llamacpp-ov.md](./contributing-llamacpp-ov.md#step-3--build-with-the-openvino-backend) for Windows and full build instructions.

### 2. Verify Backend Ops (Unit Testing)

`test-backend-ops` compares OpenVINO results against the CPU reference implementation and reports numerical divergence.

**The `-o` filter takes the short name, not the registry key:**

| Registry key | Test filter |
|---|---|
| `GGML_OP_POOL_2D` | `-o POOL_2D` |
| `GGML_UNARY_OP_RELU` | `-o RELU` |
| `GGML_GLU_OP_GEGLU_QUICK` | `-o GEGLU_QUICK` |

```bash
./build/ReleaseOV/bin/test-backend-ops -b OPENVINO -o POOL_2D
```

Ensure all tests pass without numerical divergence or runtime crashes.

**Test every device you have access to** — a device-specific limitation will not show up on CPU:

```bash
GGML_OPENVINO_DEVICE=CPU ./build/ReleaseOV/bin/test-backend-ops -b OPENVINO -o POOL_2D
GGML_OPENVINO_DEVICE=GPU ./build/ReleaseOV/bin/test-backend-ops -b OPENVINO -o POOL_2D

# NPU — keep the context small to avoid unrelated failures
GGML_OPENVINO_DEVICE=NPU ./build/ReleaseOV/bin/test-backend-ops -b OPENVINO -o POOL_2D
```

If a configuration fails only on GPU or NPU, gate it in `is_op_unsupported_case()` rather than disabling the op everywhere — see "Gating Unsupported Configurations" above.

### 3. Check the Support Table

Re-run the support command to confirm your op is registered:

```bash
./build/ReleaseOV/bin/test-backend-ops support -b OPENVINO
```

Look for your operator in the output and verify the configuration matrix reflects the support you expect.

### 4. End-to-End Model Verification (Recommended)

Download a sample model first (see [Download Sample Model](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/OPENVINO.md#3-download-sample-model)), then run inference and confirm the OpenVINO backend is actually selected — `-ngl 99` alone does not choose it:

```bash
GGML_OPENVINO_DEVICE=CPU ./build/ReleaseOV/bin/llama-simple \
    -m ~/models/Llama-3.2-1B-Instruct-Q4_0.gguf -n 50 "The story of AI is "
```

Check startup logs for the OpenVINO device line. If it's absent, your op never ran.

Also run a performance regression check (`-fa 1` is required):

```bash
./build/ReleaseOV/bin/llama-bench -m ~/models/Llama-3.2-1B-Instruct-Q4_0.gguf -fa 1
```

See [Step 5 of the contributing guide](./contributing-llamacpp-ov.md#step-5--test-your-changes) for the full GPU/NPU test matrix.

### 5. Pre-PR Checklist

- [ ] `#include <openvino/op/<op>.hpp>` added to `op_table.cpp`
- [ ] Entry in `get_supported_ops()` uses the **correct prefix** (`GGML_OP_` / `GGML_UNARY_OP_` / `GGML_GLU_OP_`)
- [ ] Custom translations declared via `GGML_OP_CONVERTER` in `op_table.h`
- [ ] `compute_op_case()` case added if the op has variants
- [ ] `is_op_unsupported_case()` case added for unsupported configs
- [ ] `compute_node_dynamic_dims()` case added if the op reshapes/permutes
- [ ] Views handled with `process_view_input_new()` where applicable
- [ ] Constants created with the input's element type
- [ ] Dimension order reversed and verified on a **non-square** shape
- [ ] Tests pass on CPU **and** GPU/NPU
- [ ] `test-backend-ops support -b OPENVINO` shows the op
- [ ] `llama-bench -fa 1` shows no performance regression
- [ ] No unrelated changes included

### 6. Open a Pull Request

**Full workflow: [contributing-llamacpp-ov.md](./contributing-llamacpp-ov.md).** Op-specific points to remember:

- **Base branch is `ravi9/llama.cpp` / `dev_backend_openvino`**, not `ggml-org/llama.cpp` master. GitHub often defaults to the wrong one — change it before submitting. Work is upstreamed after team validation.
- Use the `ggml-openvino:` commit prefix, e.g. `ggml-openvino: add support for GGML_OP_POOL_2D`.
- In the PR description, state **which devices you tested on** and which configurations you deliberately gated off in `is_op_unsupported_case()`.
- Fill in the **AI usage disclosure** field in the PR template.
- Monitor GitHub Actions, especially the OpenVINO CI jobs. Failures there often mean your mapping breaks on a different OS or hardware combination.
- Address maintainer and CI feedback with additional commits on your branch.
