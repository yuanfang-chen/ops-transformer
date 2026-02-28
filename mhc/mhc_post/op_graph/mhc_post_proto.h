/**
* @brief Fuse the main branch feature h_out with the residual branch feature x 
* using the gating mechanism h_post and the doubly stochastic matrix  h_res 
* to enable information flow.
* @par Inputs:
* @li x: A Tensor. Type is:BFloat16 or Float16.
* @li h_res: A Tensor. Type is:Float32.
* @li h_out: A Tensor. Type is:BFloat16 or Float16.
* @li h_post: A Tensor. Type is:Float32.
* @par Outputs:
* @li y: A Tensor. Type is:BFloat16 or Float16.
*/
REG_OP(MhcPost)
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(h_res, TensorType({DT_FLOAT}))
    .INPUT(h_out, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(h_post, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(MhcPost)