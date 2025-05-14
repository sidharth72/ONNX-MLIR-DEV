import numpy as np
import math


def conv_unoptimized_python(
    X_image,
    W_filter,
    B_bias,
    output_dims_info,  # A list/tuple like [N, CO, HO, WO]
    pads_info,  # A list/tuple like [pad_h_begin, pad_w_begin, pad_h_end, pad_w_end]
    # Or for N-D: [p_dim1_start, p_dim2_start, ..., p_dim1_end, p_dim2_end, ...]
    strides_info,  # A list/tuple like [stride_h, stride_w]
    dilations_info,  # A list/tuple like [dilation_h, dilation_w]
    group_info,
):
    """
    Simulates the logic of the unoptimized convolution lowering from the C++ ONNX-MLIR code.

    Args:
        X_image (np.array): The input image tensor.
                            Shape: [N, CI, HI, WI] (Batch, Channel In, Height In, Width In)
        W_filter (np.array): The filter/kernel tensor.
                             Shape: [CO, CI_per_group, KH, KW] (Channel Out, ChannelInPerGroup, KernelHeight, KernelWidth)
        B_bias (np.array or None): The bias tensor. Shape: [CO]
        output_dims_info (list/tuple): Dimensions of the output tensor [N, CO, HO, WO].
        pads_info (list/tuple): Padding values for each spatial dimension start and end.
                                Example for 2D: [pad_h_start, pad_w_start, pad_h_end, pad_w_end].
                                The C++ code often uses pads for the beginning of each dim.
                                For simplicity, we'll assume pads_info are [pad_start_dim0, pad_start_dim1, ...]
                                and calculate effective input size with padding or adjust indices.
                                Let's assume pads_info = [pad_spatial_dim0_start, pad_spatial_dim1_start, ...]
        strides_info (list/tuple): Strides for each spatial dimension. [stride_spatial_dim0, stride_spatial_dim1, ...]
        dilations_info (list/tuple): Dilations for each spatial dimension. [dilation_spatial_dim0, dilation_spatial_dim1, ...]
        group_info (int): The number of groups.
    """
    # In C++, this would be obtained from the operandAdaptor and shapeHelper
    # Operation *op = convOp.getOperation();
    # Location loc = convOp.getLoc();
    # MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, SCFBuilder,
    #     MathBuilder, MemRefBuilder>
    #     create(rewriter, loc);

    # Spatial data starts from the second dimension in ONNX [N, C, H, W]
    # so spatial_start_index for X_image would be 2.
    # For W_filter [CO, CI_per_group, KH, KW], spatial_start_index is 2.
    spatial_start_index_in_X = 2
    spatial_start_index_in_W = 2

    # auto inputOperand = operandAdaptor.getX();
    # auto filterOperand = operandAdaptor.getW();
    # auto biasOperand = operandAdaptor.getB();
    has_bias = B_bias is not None
    # int64_t groupNum = convOp.getGroup();
    G = group_info  # groupNum from C++
    # IndexExpr G_expr = LitIE(groupNum); # Symbolic expression in C++

    # Value fZero = create.math.constant(memRefType.getElementType(), 0);
    # Assuming float32 for simplicity
    element_type = np.float32
    f_zero = element_type(0.0)

    # --- Get Input Shapes (from NumPy arrays directly) ---
    N_in, CI, *HI_WI_dims = X_image.shape
    CO_out_filter, CI_per_group_filter, *KH_KW_dims = W_filter.shape

    # --- Bounds for output sizes: [N x CO x HO x WO] ---
    # int outputRank = shapeHelper.getOutputDims().size();
    output_rank = len(output_dims_info)
    # IndexExpr N = shapeHelper.getOutputDims()[0];
    N = output_dims_info[0]
    # IndexExpr CO = shapeHelper.getOutputDims()[1];
    CO = output_dims_info[1]
    # IndexExpr COPerGroup = CO.ceilDiv(G_expr);
    CO_per_group = math.ceil(CO / G)

    # --- Bounds for input image X: [N x CI x HI x WI] ---
    # (already extracted N_in, CI, HI_WI_dims)

    # --- Bounds for kernel/filter W: [CO x CIPerGroup x KH x KW] ---
    # IndexExpr CIPerGroup_expr = create.krnlIE.getShapeAsSymbol(filterOperand, 1);
    CI_per_group = W_filter.shape[1]  # This is CI_per_group_filter

    # Check if C++ CI_per_group matches filter shape
    assert CI_per_group == CI_per_group_filter, "Mismatch in CI_per_group definition"
    # Also, CI must be G * CI_per_group
    assert (
        CI == G * CI_per_group
    ), f"Input channels {CI} != groups {G} * CI_per_group {CI_per_group}"
    # And CO must be G * CO_per_group (or rather CO_per_group derived from CO and G)
    assert CO_out_filter == CO, "Output channels from filter shape mismatch expected CO"

    # --- Allocate memory for the output ---
    # This is equivalent to 'alloc' in the C++ code
    # MemRefType memRefType = mlir::cast<MemRefType>(alloc.getType());
    output_tensor_shape = tuple(output_dims_info)
    alloc = np.zeros(output_tensor_shape, dtype=element_type)

    # Determine the bounds for the loops over batch & channel out.
    # IndexExpr iZero = LitIE(0);
    # IndexExpr iOne = LitIE(1);
    i_zero = 0
    i_one = 1

    # SmallVector<Value, 3> lbsStorage, ubsStorage, stepsStorage;
    # SmallVector<IndexExpr, 3> outerLbs = {iZero, iZero, iZero};
    # SmallVector<IndexExpr, 3> outerUbs = {N, G_expr, COPerGroup_expr};
    # SmallVector<IndexExpr, 3> outerSteps = {iOne, iOne, iOne};
    # (These are for Krnl loop generation, Python directly uses range)

    # Iterate over the outer loops
    # for n_idx = 0 .. N-1:
    for n_idx in range(N):  # Loop over batch size
        #   for g_idx = 0 .. G-1:
        for g_idx in range(G):  # Loop over groups
            #     for co_per_group_idx = 0 .. COPerGroup-1:
            for co_per_group_idx in range(CO_per_group):
                #       co_idx = g_idx * COPerGroup + co_per_group_idx;
                # This is the actual output channel index
                co_idx = g_idx * CO_per_group + co_per_group_idx

                # If co_idx goes beyond CO due to ceil division for the last group, skip.
                if co_idx >= CO:
                    continue

                # IndexExprScope outerScope(create.krnl);
                # DimIndexExpr g_expr_val(outerIndices[1]);
                # DimIndexExpr coPerGroup_expr_val(outerIndices[2]);
                # IndexExpr co_expr = g_expr_val * SymIE(COPerGroup_expr) + coPerGroup_expr_val;

                # Compute g_idx * CIPerGroup for later use.
                # IndexExpr gTimesCIPerGroup_expr = g_expr_val * SymIE(CIPerGroup_expr);
                g_times_CI_per_group = g_idx * CI_per_group

                # Determine the bounds for the output spacial dimensions.
                # int spacialRank = outputRank - spatialStartIndex;
                # (spatial_start_index_in_X is 2 for NCHW format)
                spatial_rank = (
                    output_rank - spatial_start_index_in_X
                )  # e.g., 2 for 2D conv (H, W)

                # ValueRange outputSpacialLoops = create.krnl.defineLoops(spacialRank);
                # SmallVector<IndexExpr, 3> outputSpacialLbs, outputSpacialUbs;
                # for (int i = spatialStartIndex; i < outputRank; ++i) {
                #   outputSpacialLbs.emplace_back(iZero);
                #   outputSpacialUbs.emplace_back(SymIE(shapeHelper.getOutputDims()[i]));
                # }

                # --- Spatial loops (e.g., for ho, wo) ---
                # We need to generate nested loops for each spatial dimension.
                # For 2D (HO, WO):
                # for ho_idx = 0 .. HO-1:
                #   for wo_idx = 0 .. WO-1:
                # This requires a recursive helper or knowing the rank beforehand.
                # Let's assume 2D spatial for clarity (H, W). Generalizing this part:
                HO = output_dims_info[spatial_start_index_in_X]
                WO = output_dims_info[spatial_start_index_in_X + 1]  # Assuming 2D

                for ho_idx in range(HO):  # Output Height
                    for wo_idx in range(WO):  # Output Width
                        # IndexExprScope outputSpacialScope(createKrnl);
                        # MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl,
                        #     MathBuilder> create(createKrnl);

                        # ValueRange inits = ValueRange(fZero);
                        # This is the sum for the reduction (dot product)
                        current_sum = f_zero

                        # Bounds for reduction loops.
                        # ValueRange redLoops = create.krnl.defineLoops(spacialRank + 1);
                        # SmallVector<IndexExpr, 4> redLbs, redUbs, pMinOS_list;
                        # First: loop over channel in per group.
                        # redLbs.emplace_back(iZero);
                        # redUbs.emplace_back(SymIE(CIPerGroup_expr));

                        # for ci_per_group_idx = 0 .. CIPerGroup-1:
                        for ci_per_group_idx in range(CI_per_group):
                            # For each spacial dim, do the following.
                            # The C++ code calculates loop bounds for kernel (kh, kw) dynamically.
                            # This is for handling parts of the kernel that might go outside
                            # the *padded* input due to strides/dilations, effectively
                            # implementing 'VALID' like convolution for kernel loops.
                            # Here, we'll iterate fully over kernel dimensions and then
                            # calculate the input coordinates, checking bounds there.

                            # For 2D kernel (KH, KW):
                            KH = W_filter.shape[spatial_start_index_in_W]
                            KW = W_filter.shape[spatial_start_index_in_W + 1]

                            for kh_idx in range(KH):  # Kernel Height
                                for kw_idx in range(KW):  # Kernel Width
                                    # IndexExprScope redScope(createKrnl);
                                    # MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl,
                                    #     MathBuilder> create(createKrnl);

                                    # --- Calculate input coordinates ---
                                    # Create access function for input image:
                                    # input_access_indices: [n_idx, ci, input_h, input_w].
                                    # ci = g_idx * CI_per_group + ci_per_group_idx
                                    actual_ci_idx = (
                                        g_times_CI_per_group + ci_per_group_idx
                                    )

                                    # for each spacial dim: access is o * s + k * d - p.
                                    # DimIndexExpr k(redIndices[1 + i]);
                                    # SymbolIndexExpr pos(pMinOS[i]); -> p - o * s
                                    # LiteralIndexExpr d(shapeHelper.dilations[i]);
                                    # IndexExpr t = (k * d) - pos; -> k*d - (p - o*s) = k*d - p + o*s

                                    # Input spatial coordinates
                                    # For H dimension:
                                    # H_in_coord = ho_idx * stride_h + kh_idx * dilation_h - pad_h_start
                                    h_in_coord = (
                                        ho_idx * strides_info[0]
                                        + kh_idx * dilations_info[0]
                                        - pads_info[0]
                                    )
                                    # For W dimension:
                                    # W_in_coord = wo_idx * stride_w + kw_idx * dilation_w - pad_w_start
                                    w_in_coord = (
                                        wo_idx * strides_info[1]
                                        + kw_idx * dilations_info[1]
                                        - pads_info[1]
                                    )  # Assuming pads_info[1] is pad_w_start

                                    image_pixel_value = f_zero
                                    # Boundary check for input spatial coordinates
                                    if (
                                        h_in_coord >= 0
                                        and h_in_coord
                                        < X_image.shape[spatial_start_index_in_X]
                                        and w_in_coord >= 0
                                        and w_in_coord
                                        < X_image.shape[spatial_start_index_in_X + 1]
                                    ):
                                        # Value image = create.krnl.loadIE(inputOperand, inputAccessFct);
                                        image_pixel_value = X_image[
                                            n_idx, actual_ci_idx, h_in_coord, w_in_coord
                                        ]

                                    # Create access fct for filter: [co_idx, ci_per_group_idx, kh_idx, kw_idx].
                                    # SmallVector<IndexExpr, 4> filterAccessFct;
                                    # filterAccessFct.emplace_back(DimIE(co_expr_val));
                                    # filterAccessFct.emplace_back(DimIE(ciPerG_expr_val));
                                    # for (int i = 0; i < spacialRank; ++i) {
                                    #   DimIndexExpr k(redIndices[1 + i]);
                                    #   filterAccessFct.emplace_back(k);
                                    # }
                                    # Value filter_val = create.krnl.loadIE(filterOperand, filterAccessFct);
                                    filter_pixel_value = W_filter[
                                        co_idx, ci_per_group_idx, kh_idx, kw_idx
                                    ]

                                    # Value oldRed = iterArg; (current_sum in Python)
                                    # Value mul = create.math.mul(image, filter_val);
                                    multiplied_value = (
                                        image_pixel_value * filter_pixel_value
                                    )
                                    # Value newRed = create.math.add(oldRed, mul);
                                    current_sum += multiplied_value
                                    # create.krnl.yield(newRed); (Implicit in Python loop)
                                # End of reduction loops for kw
                            # End of reduction loops for kh
                        # End of reduction loops for ci_per_group

                        # Value result = innerIterate.getResult(0); (current_sum in Python)
                        # Store the result. Optionally add bias.
                        # SymbolIndexExpr coInOutputSpacial(co_expr_val);
                        if has_bias:
                            # Value bias_val = create.krnl.loadIE(biasOperand, {coInOutputSpacial});
                            bias_for_channel = B_bias[co_idx]
                            # result = create.math.add(result, bias_val);
                            current_sum += bias_for_channel

                        # SmallVector<IndexExpr, 4> resAccessFunc;
                        # resAccessFunc.emplace_back(SymIE(outerIndices[0])); -> n_idx
                        # resAccessFunc.emplace_back(coInOutputSpacial); -> co_idx
                        # for (Value o : outputSpatialIndices)
                        #   resAccessFunc.emplace_back(DimIE(o)); -> ho_idx, wo_idx
                        # create.krnl.storeIE(result, alloc, resAccessFunc);
                        alloc[n_idx, co_idx, ho_idx, wo_idx] = current_sum
                    # End of output spatial loop for wo_idx
                # End of output spatial loop for ho_idx
            # End of co_per_group_idx loop
        # End of g_idx loop
    # End of n_idx loop

    # rewriter.replaceOp(op, alloc);
    # onnxToKrnlSimdReport(op);
    # return success();
    return alloc


if __name__ == "__main__":
    # --- Example Usage (matches a simple ONNX Conv scenario) ---
    print("Simulating Convolution Lowering Logic in Python...")

    # --- Configuration ---
    N_val = 1  # Batch size
    CI_val = 1  # Input Channels
    HI_val = 5  # Input Height
    WI_val = 5  # Input Width

    CO_val = 1  # Output Channels
    KH_val = 3  # Kernel Height
    KW_val = 3  # Kernel Width

    group_val = 1
    # CI_per_group_val = CI_val // group_val (must hold)
    CI_per_group_val = CI_val // group_val

    strides_val = [1, 1]  # Stride H, Stride W
    pads_val = [0, 0, 0, 0]  # Pad H_start, W_start, H_end, W_end
    # For this Python code, we'll use just [pad_H_start, pad_W_start]
    effective_pads_starts = [pads_val[0], pads_val[1]]

    dilations_val = [1, 1]  # Dilation H, Dilation W

    # Calculate Output Dimensions (HO, WO)
    # HO = floor((HI + pad_h_start + pad_h_end - ((KH-1)*dilation_h + 1)) / stride_h) + 1
    HO_val = (
        math.floor(
            (HI_val + pads_val[0] + pads_val[2] - ((KH_val - 1) * dilations_val[0] + 1))
            / strides_val[0]
        )
        + 1
    )
    WO_val = (
        math.floor(
            (WI_val + pads_val[1] + pads_val[3] - ((KW_val - 1) * dilations_val[1] + 1))
            / strides_val[1]
        )
        + 1
    )

    output_dims = [N_val, CO_val, HO_val, WO_val]

    # --- Create Tensors ---
    # Input Image (e.g., a 5x5 image with one channel)
    X = np.arange(N_val * CI_val * HI_val * WI_val, dtype=np.float32).reshape(
        (N_val, CI_val, HI_val, WI_val)
    )
    # X = np.ones((N_val, CI_val, HI_val, WI_val), dtype=np.float32) # Simpler input
    print("Input X:\n", X)

    # Filter (e.g., a 3x3 filter)
    W = np.ones((CO_val, CI_per_group_val, KH_val, KW_val), dtype=np.float32)
    # W[0, 0, 1, 1] = 2 # Make one weight different
    print("\nFilter W:\n", W)

    # Bias (optional)
    B = np.array([0.5], dtype=np.float32)  # Bias for each output channel
    # B = None
    print("\nBias B:\n", B)

    print(f"\nOutput dimensions: {output_dims}")
    print(f"Pads (starts): {effective_pads_starts}")
    print(f"Strides: {strides_val}")
    print(f"Dilations: {dilations_val}")
    print(f"Groups: {group_val}")

    # --- Perform Convolution ---
    output_result = conv_unoptimized_python(
        X,
        W,
        B,
        output_dims,
        effective_pads_starts,  # Using only start pads for simplicity in this Python version's indexing
        strides_val,
        dilations_val,
        group_val,
    )

    print("\nOutput Result (from Python simulation):\n", output_result)

    # --- Verification with a library (optional, e.g., PyTorch or TensorFlow) ---
    # This is to check if our manual calculation is correct.
    try:
        import torch
        import torch.nn.functional as F

        # PyTorch expects input as (N, C_in, H_in, W_in)
        # PyTorch expects weights as (C_out, C_in/groups, kH, kW)
        X_torch = torch.from_numpy(X)
        W_torch = torch.from_numpy(W)
        B_torch = torch.from_numpy(B) if B is not None else None

        # Note: PyTorch padding is [pad_w_start, pad_w_end, pad_h_start, pad_h_end] or just (pad_h, pad_w)
        # Our pads_val is [pad_h_start, pad_w_start, pad_h_end, pad_w_end]
        # So for PyTorch, if symmetric padding, it's (pads_val[0], pads_val[1])
        # If asymmetric, it's more complex. Let's assume symmetric for this example if pads_val[0]==pads_val[2]
        # For this example, pads_val is [0,0,0,0] so PyTorch padding is (0,0)
        torch_padding = (
            pads_val[0],
            pads_val[1],
        )  # (pad_H, pad_W) assuming symmetric or just start pads

        output_torch = F.conv2d(
            X_torch,
            W_torch,
            bias=B_torch,
            stride=strides_val,
            padding=torch_padding,  # PyTorch padding can be int or tuple
            dilation=dilations_val,
            groups=group_val,
        )
        print(
            "\nOutput Result (from PyTorch for verification):\n", output_torch.numpy()
        )

        assert np.allclose(
            output_result, output_torch.numpy(), atol=1e-5
        ), "Mismatch with PyTorch!"
        print("\nVerification with PyTorch: SUCCESSFUL!")

    except ImportError:
        print("\nPyTorch not installed, skipping verification step.")
    except Exception as e:
        print(f"\nError during PyTorch verification: {e}")
