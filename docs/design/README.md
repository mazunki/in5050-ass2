

# Desgin Home Exam: Video Encoding on the nVIDA Tegra Xavier Heterogeneous SoC

## memory
- remove `memcpy` 
- cache multiplicaitions
- avoid interleaving memory by making sure memory reads/writes are sequential
- fit memory into aligned memory chunks

## SIMD
- flatten for loops in dct/idct
- make use of the ALU buffers by exploiting the nature of DCO by saturating the cachelines inside each for-loop

## threading 
- run `write_frame()` on a background pthread to avoid disk stalling (maybe push into an out-queue/framebuffer)
- run each call to `{dct_quantize,dequantize_idct}_row` on a separate thread since the out_data is independent
- iterate over the macroblocks in the row function on separate threads since they're mutually exclusive