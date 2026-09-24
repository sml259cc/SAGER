# Method implementation

## Computation

1. **Text and audio encoding.** Frozen RoBERTa CLS features and data2vec audio
   tokens are projected into a shared representation. Bidirectional dialogue and
   speaker memory, sparse temporal convolutional experts, and a BiGRU provide
   contextual states. Audio temporal routing also uses quality cues.
2. **Cross-modal and graph evidence.** Text queries attend to audio keys and values
   through two attention layers. Stable Graph propagates context while retaining
   initial features. Typed Graph computes relation-specific neighborhood summaries
   and differences from those summaries, with learned relation and quality gates.
3. **Utility and evidence routing.** A scalar tanh head predicts audio utility.
   Candidate audio, cross-modal, low-band, and high-band logits are combined with
   masked routing weights to form base evidence `z_B`.
4. **Relational consensus.** Local, Speaker, and Context paths produce three
   proposals. Their weights use path state, proposal entropy/disagreement,
   boundary, quality, and audio-utility cues. Their weighted sum is `z_C`.
5. **Verification and adoption.** The verifier decides how much consensus to
   incorporate into the base evidence. The adoption gate controls revision of
   frozen text-anchor logits `z_T`.

The two interpolations are:

```text
z_E = z_B + s_ver * v * (z_C - z_B)
z   = z_T + g * (z_E - z_T)
```

Cross-attention is scaled as:

```text
H_cross = H_text + s_ca * (F_ca(H_text, H_audio) - H_text)
```

`s_ca` and `s_ver` are global learned strengths initialized to 1 and projected
onto `[0,1]` after each optimizer step. The utterance-dependent `v` and `g` are
sigmoid outputs. If no evidence is available, the adoption gate is zero and the
output retains the text anchor. The model uses text and audio only.

## Predictive-benefit supervision

For dialogue-grouped OOF ridge logits `xi_T` and `xi_TA`, cross-entropy `CE`,
and correctness indicator `correct`, the audio utility target is:

```text
u_target = clip(
    0.25 * clip(CE(xi_T, y) - CE(xi_TA, y), -2, 2)
    + 0.5 * (correct(xi_TA, y) - correct(xi_T, y)),
    -1, 1
)
```

Mean Huber loss supervises predicted audio utility on valid audio targets.
Verifier targets compare the cross-entropies of consensus and base evidence;
adoption targets compare evidence and anchor cross-entropies. Comparison logits
are detached for target construction. Balanced binary cross-entropy supervises
both decisions, with adoption supervision restricted to available evidence.

The total objective combines final focal classification, audio utility loss,
verifier and adoption losses, and auxiliary classification, boundary,
representation, prediction-consistency, and expert-balancing terms. Their
coefficients and other implementation settings are explicit in the YAML files.
Inference uses the learned network without labels or ridge predictors.

## Code mapping

| Component | Implementation |
|---|---|
| Model interface and frozen inputs | `model/sager.py` |
| Projection, temporal experts, cross-attention, auxiliary losses | `model/components.py` |
| Memory, graphs, routing, consensus, verification, adoption | `model/backbone.py` |
| Two global strengths and projection | `model/module_strength.py` |
| Dialogue-grouped OOF utility targets | `preprocessing/utility_targets.py` |
| Feature loading and dialogue batches | `utils/dataloader.py` |
| Optimization and development selection | `train.py` |
| Checkpoint-bound evaluation | `evaluate.py` |

The internal parameter names `cross_attention` and `relation_path` correspond
to `s_ca` and `s_ver`, respectively. The model accepts the prepared input schema
in [Data preparation](data.md). Each utterance supplies one frozen CLS vector and has one predicted audio utility.
