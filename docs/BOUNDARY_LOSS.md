# E1 signed-distance Boundary Loss

For binary mask M, let `P` be foreground and `N` its complement. The unnormalized map is

`phi = EDT(N)*N - (EDT(P)-1)*P`.

Thus background distances are positive, foreground interior distances are negative, and the foreground-side boundary is zero, matching the reference Kervadec implementation convention. For a nonempty mask, `phi_norm = phi / max(abs(phi))`, giving a fixed per-image range within [-1,1]. For an empty mask, `phi_norm=0`; this explicitly avoids undefined foreground distances and follows the reference implementation's zero-map treatment for an absent class.

The differentiable term is `L_boundary = mean(sigmoid(z) * phi_norm)` over batch, channel, height, and width. E1 optimizes `L_BCE + L_Dice + L_FT + 0.1 L_boundary`. It never thresholds the prediction inside the loss. The initial base-loss and weighted-boundary gradient L2 norms are saved for every run before epoch 1.

For the fixed seed-42 initial batch, the base loss was 2.15443, the unweighted boundary term was 0.05364, the base gradient L2 norm was 10.97982, and the lambda-weighted boundary gradient L2 norm was 0.21855 (about 1.99% of the base norm). This was not judged a severe scale collapse, so the preregistered lambda=0.1 was retained and lambda=1.0 was not tried.
