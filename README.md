# IBM Research MRE

A collection of Minimal Reproducible Examples (MRE) for debugging purposes with relation to the IBM VXE/VXE Vector Intrinsics or the zDNN/NNPA Library.

### Why ARM NEON References?

Most of my work done on Llama.cpp was referenced against ARM NEON's vector intrinsics because ARM NEON uses 128-bit vector width. It is also easier to correlate ARM NEON's vector intrinsics towards s390x's VXE/VXE2 vector intrinsics.
