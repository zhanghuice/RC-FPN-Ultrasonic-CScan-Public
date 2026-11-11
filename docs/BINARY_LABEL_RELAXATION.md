# E2 binary boundary-label relaxation

For original binary target M, the relaxed band is `B = (dilate_3x3(M) != erode_3x3(M))`. Outside B the only allowed class is the original class. Within B both binary classes `{0,1}` are allowed, so the binary BCE contribution degenerates to zero.

The implementation uses unreduced BCE and the global valid-pixel reduction

`L_BCE_relaxed = sum((1-B) * BCEWithLogits(z,M)) / max(sum(1-B),1)`.

Dice and Focal Tversky are not relaxed; both use original M with exactly the E0 implementation and parameters. Therefore E2 is a Zhu-inspired binary/joint-loss adaptation only. It is neither global label smoothing nor a reproduction of video propagation and the full multiclass model.
