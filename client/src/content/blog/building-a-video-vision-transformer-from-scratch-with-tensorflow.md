## Building a Video Vision Transformer from Scratch with TensorFlow

The Transformer architecture, initially a revolutionary concept in Natural Language Processing (NLP), has dramatically reshaped the field of Computer Vision. The Vision Transformer (ViT) showed that transformers could achieve state-of-the-art results on image tasks by treating images as sequences of patches.

But what about videos? Videos add a temporal dimension, making the task more complex. How can we adapt the powerful ViT to understand action and motion?

In this post, we'll dive into the code of a custom Vision Transformer implementation for **video classification**, built using TensorFlow and Keras. We'll break down each component, from turning video frames into "tubelets" to the final classification head, explaining the core concepts along the way.

The code we'll be exploring is available [Github Repo](https://github.com/Abdulrahman-Adel/Real-Life-Violence-Detection).

### The Challenge: From Images to Videos

Traditional convolutional networks process images (2D spatial data). Videos add a third dimension: time. A video is essentially a sequence of images. To apply a transformer, which works with sequences, we need a way to linearize the spatio-temporal information of a video.

For static images, ViT splits the image into patches. For videos, a natural extension is to split the video into **spatio-temporal patches**, often referred to as **Tubelets**. Imagine taking small 3D cubes (width x height x time) from the video volume.

### High-Level Architecture

Our Video Vision Transformer follows a similar structure to its image counterpart, adapted for the temporal dimension:

1.  **Input Video:** A batch of videos, each represented as a 5D tensor `[batch_size, Depth, Height, Width, Channels]`.
2.  **Tubelet Embedding:** Extract spatio-temporal patches (tubelets) and project them into a high-dimensional embedding space.
3.  **Positional Encoding:** Add learnable positional information to the patch embeddings so the model knows where each tubelet came from in the original video volume. A learnable "classification token" is also prepended.
4.  **Transformer Encoder:** A stack of transformer blocks processes the sequence of embedded tubelets. Each block performs Multi-Head Self-Attention and passes the result through a Feed-Forward Network (MLP), using Layer Normalization and residual connections.
5.  **Classification Head:** The final representation (derived from the output sequence) is passed through a linear layer for classification.


Let's break down each part as implemented in the code.

### 1. Tubelet Embedding (`Projection` Layer)

This is the first crucial step: converting the raw video tensor into a sequence of embedded tubelets. Your `Projection` layer handles this using a `Conv3D` layer.

```python
class Projection(keras.layers.Layer):
    """
    Custom layer for dividing video frames into patches and implementing linear projection step
    using Tubelet Embedding.

    * Input
    * Conv3D
    * Reshape

    Parameters
    =========

    img_size: int
        Here is assumed that the video hight, width and depth are of the same size

    embed_size: int
        Embedding size.

    patch_size: int
        Size of patch the images with be split into.

    Inputs
    =======

    input shape: shape
        [batch_size, Depth, Hight, Width, Channels]

    Outputs
    =======

    Output shape: shape
        [batch_size, n_patches, embed_size]
    """

    def __init__(self, embed_size, patch_size, img_size, **kwargs):
        super().__init__(**kwargs)
        self.n_patches = 0
        self.patch_size = patch_size
        self.conv = keras.layers.Conv3D(
            filters=embed_size,
            kernel_size=(patch_size, patch_size, 2),
            strides=(patch_size, patch_size, 2),
            padding="VALID",
        )
        self.reshape = keras.layers.Reshape(target_shape=(-1, embed_size))

    def build(self, input_shape):
        bz, d, h, w, c = input_shape
        self.n_patches = (
            d * h * w // self.patch_size**3
        )  # (n_patches) = Hight * width * Depth / (Patch size) ^3

    def call(self, videos):
        x = videos
        x = self.conv(
            x
        )  # shape --> [batch_size, n_patches ** (3/2), n_patches ** (3/2), n_patches ** (3/2), embed_size]

        return self.reshape(x)
```

**Explanation:**

* Instead of 2D convolutions for image patches, we use `Conv3D`. This kernel slides through the video volume (Height, Width, Depth).
* `kernel_size=(patch_size, patch_size, 2)` means each "tubelet" processed by the convolution will cover `patch_size` pixels spatially (H x W) and `2` frames temporally.
* `strides=(patch_size, patch_size, 2)` means the kernel moves `patch_size` pixels horizontally and vertically, and skips 1 frame (moves 2 frames) temporally. This effectively extracts non-overlapping or partially overlapping tubelets.
* The output of the `Conv3D` has reduced spatial and temporal dimensions but has `embed_size` channels.
* The `Reshape` layer flattens the spatial and temporal dimensions into a single "sequence length" dimension, resulting in a tensor of shape `[batch_size, number_of_tubelets, embed_size]`.

### 2. Class Token and Positional Embeddings

Like the standard ViT, we prepend a learnable `[CLS]` token to the sequence. This token will interact with all tubelet embeddings through self-attention and its final representation is typically used for the classification task.

Since the transformer is permutation-invariant (it doesn't naturally understand the order or position of tokens), we add learned **positional embeddings** to the sequence of tubelet embeddings and the `[CLS]` token. These embeddings are trained alongside the rest of the model and provide location information.

```python
class VisionTransformer(keras.Model):
    # ... (init method)

    def call(self, inputs):
        # ... (initial parts)

        linear_embed = self.proj(inputs) # shape --> [batch_size, n_patches, embed_size]

        # Prepend the learnable CLS token
        # The cls_token variable has shape [1, 1, embed_size]
        # We need to broadcast it to match the batch size of the input video
        broadcast_shape = tf.where(
            [True, False, False], # Broadcast along the first dimension (batch size)
            keras.backend.shape(tf.expand_dims(linear_embed[:, 0], axis=1)), # Get batch size from linear_embed
            [0, 1, self.embed_size], # Keep the last two dimensions as they are
        )
        cls_token = tf.broadcast_to(self.cls_token, shape=broadcast_shape)
        # cls_token shape: [batch_size, 1, embed_size]

        # Concatenate the CLS token with the tubelet embeddings
        linear_proj = tf.concat(
            [cls_token, linear_embed], axis=1
        )  # shape --> [batch_size, n_patches + 1, embed_size]

        # Add learnable positional embeddings
        # self.pos_embed has shape [1, n_patches + 1, embed_size]
        # TensorFlow's broadcasting handles adding this to each item in the batch
        linear_proj = linear_proj + self.pos_embed # shape --> [batch_size, n_patches + 1, embed_size]

        # Apply dropout to the embedded sequence
        x = self.drop(linear_proj)

        # ... (rest of the call method)
```

**Explanation:**

* `self.cls_token` is a trainable variable initialized with zeros. Its shape is `[1, 1, embed_size]`.
* We use `tf.broadcast_to` to make the `cls_token` match the batch size of the input video, resulting in shape `[batch_size, 1, embed_size]`.
* `tf.concat` joins the `cls_token` and the `linear_embed` (tubelet embeddings) along `axis=1` (the sequence length dimension). The sequence now has length `n_patches + 1`.
* `self.pos_embed` is another trainable variable, initialized to zeros, with shape `[1, n_patches + 1, embed_size]`. It holds the learned positional information for each position in the sequence (including the `[CLS]` token position).
* Adding `self.pos_embed` to `linear_proj` imbues each token embedding with its positional context.
* Dropout is applied to the combined embeddings before they enter the transformer blocks.

### 3. Transformer Encoder Blocks

The core of the model is a stack of identical Transformer Encoder blocks. Each block consists of:

* Layer Normalization
* Multi-Head Self-Attention (MHA)
* Residual Connection
* Layer Normalization
* Multi-Layer Perceptron (MLP)
* Residual Connection

Your code implements these as separate layers: `MHA`, `MLP`, and `TransformerEncoder`.

#### 3.1 Multi-Head Self-Attention (`MHA`)

Self-attention allows each token in the sequence (each tubelet embedding and the CLS token) to attend to all other tokens in the sequence, learning relationships between them. Multi-head attention does this multiple times in parallel with different linear projections, allowing the model to jointly attend to information from different representation subspaces.

```python
class MHA(keras.layers.Layer):
    """
    Class Implementing multi-head self attention

    * head = softmax(q @ k_t // scale) @ v
    * concat(heads)
    * Linear Projection

    parameters
    ==========

    embed_dim: int
        Embedding size.

    n_head: int
        Number of heads of the multi-head self attention.

    Input
    =====

    input shape: shape
        [batch_size, n_patches + 1, embed_size]

    Output
    =====

    output shape: shape
        [batch_size, n_patches + 1, embed_size]
    """

    def __init__(self, embed_size, n_heads, dropout_rate):
        super(MHA, self).__init__()

        self.n_heads = n_heads
        self.head_dim = (
            embed_size // n_heads
        )  # when concatenated will result in embed_size
        self.scale = self.head_dim ** (-0.5)

        self.query = keras.layers.Dense(self.head_dim)
        self.key = keras.layers.Dense(self.head_dim)
        self.value = keras.layers.Dense(self.head_dim)
        self.softmax = keras.layers.Softmax()
        self.drop1 = keras.layers.Dropout(dropout_rate)

        self.proj = keras.layers.Dense(embed_size)
        self.drop2 = keras.layers.Dropout(dropout_rate)

    def call(self, inputs):
        q, k, v = self.query(inputs), self.key(inputs), self.value(inputs)
        k_t = tf.transpose(
            k, perm=[0, 2, 1]
        )  # Transpose --> [batch_size, head_dim, n_patches + 1]

        attn_filter = (q @ k_t) * self.scale
        attn_filter = self.drop1(self.softmax(attn_filter))

        attn_head = attn_filter @ v
        attn_head = tf.expand_dims(
            attn_head, axis=0
        )  # [1, batch_size, n_patches + 1, head_dim]

        heads = tf.concat(
            [attn_head for _ in range(self.n_heads)], axis=0
        )  # [n_heads, batch_size, n_patches + 1, head_dim]
        heads = tf.transpose(
            heads, perm=[1, 2, 3, 0]
        )  # [batch_size, n_patches + 1, head_dim, n_heads]

        bs, n_p, hd, nh = [keras.backend.shape(heads)[k] for k in range(4)]
        heads = tf.reshape(
            heads, [bs, n_p, hd * nh]
        )  # [batch_size, n_patches + 1, embed_dim]

        return self.drop2(self.proj(heads))
```

**Explanation:**

* The input sequence is linearly projected into Query ($Q$), Key ($K$), and Value ($V$) tensors. In this specific implementation, these initial projections are done to `head_dim`.
* Self-attention scores are calculated by multiplying $Q$ and the transpose of $K$ ($Q K^T$).
* The scores are scaled by the square root of the key dimension ($\sqrt{d_k}$, which is `self.scale`) to prevent large values that could lead to vanishing gradients after softmax.
* A softmax is applied to the scaled scores to get attention weights, indicating how much each token should attend to every other token.
* The attention weights are multiplied by the $V$ tensor to get the output for a *single* attention head.
* The result, shaped `[batch_size, n_patches + 1, embed_size]`, is passed through a final linear projection (`self.proj`).
* Dropout is applied at different stages for regularization.


#### 3.2 Feed-Forward Network (`MLP`)

After the attention mechanism, each token's representation is independently passed through a simple two-layer Feed-Forward Network (MLP).

```python
class MLP(keras.layers.Layer):
    """
    Class Implementing FeedForward Layer.

    * Linear
    * Activation (GELU)
    * Linear

    parameters
    ==========

    embed_size: int
        Embedding size.

    hidden_size: int
        output dim of first hidden layer

    activation_fn: str
        activation function applied after the first hidden layer

    Input
    =====

    input shape: shape
        [batch_size, n_patches + 1, embed_size]

    Output
    =====

    output shape: shape
        [batch_size, n_patches + 1, embed_size]
    """

    def __init__(self, embed_size, hidden_size, activation_fn="gelu", dropout_rate=0.2):
        super(MLP, self).__init__()

        self.Hidden = keras.layers.Dense(hidden_size)
        self.drop1 = keras.layers.Dropout(dropout_rate)
        self.activation = keras.activations.get(activation_fn)

        self.Linear = keras.layers.Dense(embed_size)
        self.drop2 = keras.layers.Dropout(dropout_rate)

    def call(self, inputs):
        x = inputs
        x = self.Hidden(x)
        x = self.drop1(self.activation(x))

        return self.drop2(self.Linear(x))
```

**Explanation:**

* The input is first projected to a higher dimension (`hidden_size`), typically 4 times the `embed_size` (as seen in the `if __name__ == "__main__":` block: `mlpHidden_size=198 * 4`).
* An activation function (commonly GELU in transformers) is applied.
* Dropout is applied for regularization.
* The result is then projected back to the original `embed_size`.
* This MLP is applied position-wise and independently to each token in the sequence.

#### 3.3 Transformer Encoder Block (`TransformerEncoder`)

The `TransformerEncoder` layer orchestrates the MHA and MLP sub-layers, incorporating Layer Normalization and residual connections.

```python
class TransformerEncoder(keras.layers.Layer):
    """
    Class for implementing Transformer Encoder Block.

    * Input
    * LayerNorm
    * Multi-head self attention
    * residual connection
    * LayerNorm
    * Multi-layer perceptron
    * residual connection

    parameters
    ==========
    # ... (parameters)

    Input
    =====

    input shape: shape
        [batch_size, n_patches + 1, embed_size]

    Output
    =====

    output shape: shape
        [batch_size, n_patches + 1, embed_size]
    """

    def __init__(
        self,
        embed_size,
        n_heads,
        mlpHidden_size,
        mlp_activation,
        mlp_dropout,
        attn_dropout,
    ):
        super(TransformerEncoder, self).__init__()

        # Layer Normalization applied BEFORE the sub-layer
        self.norm1 = keras.layers.LayerNormalization(epsilon=1e-3)
        self.MHA = MHA(embed_size, n_heads, attn_dropout)
        self.MLP = MLP(embed_size, mlpHidden_size, mlp_activation, mlp_dropout)
        # Layer Normalization applied BEFORE the second sub-layer
        self.norm2 = keras.layers.LayerNormalization(epsilon=1e-3)

    def call(self, inputs):
        x = inputs

        # First block: LayerNorm -> MHA -> Residual Connection
        normalized_x = self.norm1(x)
        attention_output = self.MHA(normalized_x)
        x = x + attention_output # Add the input (residual connection)

        # Second block: LayerNorm -> MLP -> Residual Connection
        normalized_x = self.norm2(x)
        mlp_output = self.MLP(normalized_x)
        y = x + mlp_output # Add the output of the first block (residual connection)

        return y
```

**Explanation:**

* Layer Normalization is applied before both the MHA and MLP sub-layers. This helps stabilize training. It normalizes the activations across the feature dimension for each token independently.
* **Residual Connections:** The output of each sub-layer (MHA and MLP) is added back to its input (`x = x + attention_output`, `y = x + mlp_output`). This allows gradients to flow more easily through the network, helping with the training of deep models. This is often called an "Add & Norm" block, though your code applies Norm *before* Add. (Pre-LayerNorm is a common and often preferred variation).
* Multiple `TransformerEncoder` blocks are typically stacked sequentially (as done in your `VisionTransformer` class using `keras.Sequential`) to build a deep transformer model.

### 4. Classification Head

After passing through the stack of transformer encoder blocks, the sequence of processed token embeddings is fed to the classification head.

```python
class VisionTransformer(keras.Model):
    # ... (init and other methods)

    def call(self, inputs):
        # ... (Projection, CLS token, Positional Embedding, Dropout, Encoder_blocks)

        x = self.Encoder_blocks(self.drop(linear_proj)) # Output of the transformer stack
        x = self.norm(x) # Final Layer Normalization

        # Classification based on the sequence representation
        # Note: GlobalAvgPool1D pools across the sequence length dimension (n_patches + 1)
        # This averages the features of all tokens (including the CLS token)
        # Standard ViT often uses only the output of the CLS token (index 0) for classification.
        # self.pooling = keras.layers.GlobalAvgPool1D()
        cls_token_final = self.pooling(x) # shape --> [batch_size, embed_size] (after pooling across sequence)

        # Final Linear layer for classification
        # Assuming n_classes=2 based on the example, sigmoid activation is appropriate
        self.Linear = keras.layers.Dense(1, activation="sigmoid") # Output dimension is 1 for binary classification

        return self.Linear(cls_token_final) # shape --> [batch_size, 1] for binary

# In __main__, n_classes is set to 2, but the final dense layer has 1 output unit with sigmoid.
# This is suitable for binary classification (2 classes). For multi-class > 2,
# the final layer should have `n_classes` units and typically 'softmax' activation.
# If using n_classes > 2 with 1 output unit + sigmoid, it's likely treated as
# predicting a single regression value or using a different loss setup.
# Let's assume binary classification based on the code example.
```

**Explanation:**

* A final `LayerNormalization` is applied to the output of the transformer stack.
* `keras.layers.GlobalAvgPool1D()` is used to pool the features across the sequence length dimension (`n_patches + 1`). This averages the embeddings of *all* tubelets and the `[CLS]` token. This differs slightly from the original ViT paper which typically uses only the output embedding of the `[CLS]` token for classification. Averaging across all patches can also be a valid approach, summarizing information from the entire video volume.
* A final `keras.layers.Dense` layer maps the pooled representation (`embed_size`) to the number of output classes (`n_classes`). For binary classification (`n_classes=2` in the example, but 1 output unit), a sigmoid activation is used to output a probability. For multi-class classification ($>2$ classes), the output dimension would be `n_classes` and the activation would typically be 'softmax'.

### Putting It All Together (`VisionTransformer`)

The main `VisionTransformer` class orchestrates the flow by initializing and connecting the layers described above.

```python
class VisionTransformer(keras.Model):
    """
    Class for implementing Vision Transformer Architecture for Video Classification.

    * Input
    * Linear Projection (Tubelet Embedding)
    * Prepend cls token then add positional embedding
    * Transformer encoder blocks (stack)
    * LayerNorm
    * Pooling (Global Average)
    * Linear classification head

    parameters
    ==========
    # ... (parameters)
    """

    def __init__(
        self,
        n_heads: int = 12,
        n_blocks: int = 12,
        img_size: int = 224, # Not directly used in Projection's kernel/stride calculation but affects input shape
        n_classes: int = 100, # Used to determine output shape of final Linear layer
        patch_size: int = 16, # Used for spatial kernel/stride in Projection
        embed_size: int = 786,
        mlp_dropout: float = 0.0,
        pos_dropout: float = 0.0,
        attn_dropout: float = 0.0,
        mlpHidden_size: int = 3072,
        mlp_activation: str = "gelu",
        **kwargs
    ):
        super(VisionTransformer, self).__init__(**kwargs)

        self.embed_size = embed_size
        # Tubelet Embedding Layer
        self.proj = Projection(embed_size, patch_size, img_size)

        # Learnable CLS token and Positional Embeddings
        # pos_embed shape depends on the number of patches, which is determined by Projection's build method
        self.cls_token = tf.Variable(tf.zeros(shape=[1, 1, embed_size]))
        # pos_embed will be created in build after proj's n_patches is known
        # self.pos_embed = tf.Variable(tf.zeros(shape=[1, self.proj.n_patches + 1, embed_size])) # Cannot access self.proj.n_patches here directly
        self.drop = keras.layers.Dropout(pos_dropout)

        # Stack of Transformer Encoder Blocks
        self.Encoder_blocks = keras.Sequential(
            [
                TransformerEncoder(
                    embed_size,
                    n_heads,
                    mlpHidden_size,
                    mlp_activation,
                    mlp_dropout,
                    attn_dropout,
                )
                for _ in range(n_blocks)
            ]
        )

        # Final Layers
        self.norm = keras.layers.LayerNormalization(epsilon=1e-6)
        # Pooling layer to get fixed-size representation
        self.pooling = keras.layers.GlobalAvgPool1D() # Pools across the sequence dimension
        # Classification Layer - adjusted to use n_classes parameter
        # Use sigmoid for n_classes=2, softmax for n_classes > 2
        if n_classes == 2:
             self.Linear = keras.layers.Dense(1, activation="sigmoid")
        else:
             self.Linear = keras.layers.Dense(n_classes, activation="softmax")


    def build(self, input_shape):
         # Call build of the Projection layer first to determine n_patches
         self.proj.build(input_shape)
         # Now we know n_patches, so we can initialize pos_embed
         self.pos_embed = tf.Variable(tf.zeros(shape=[1, self.proj.n_patches + 1, self.embed_size]), trainable=True)
         super().build(input_shape) # Complete the build process


    def call(self, inputs):
        # Ensure pos_embed is built (important if calling model without explicit build)
        if not hasattr(self, 'pos_embed'):
            self.build(inputs.shape)

        batch_size, depth, hight, width, channel = inputs.shape

        # 1. Tubelet Embedding
        linear_embed = self.proj(inputs) # shape --> [batch_size, n_patches, embed_size]

        # 2. Prepend CLS token and add Positional Embeddings
        broadcast_shape = tf.where(
            [True, False, False],
            keras.backend.shape(tf.expand_dims(linear_embed[:, 0], axis=1)),
            [0, 1, self.embed_size],
        )
        cls_token = tf.broadcast_to(self.cls_token, shape=broadcast_shape)

        linear_proj = tf.concat(
            [cls_token, linear_embed], axis=1
        )  # shape --> [batch_size, n_patches + 1, embed_size]
        linear_proj = linear_proj + self.pos_embed # Add positional embeddings

        x = self.drop(linear_proj) # Apply positional dropout

        # 3. Transformer Encoder Blocks
        x = self.Encoder_blocks(x) # shape --> [batch_size, n_patches + 1, embed_size]

        # 4. Classification Head
        x = self.norm(x) # Final Layer Normalization

        # Pool across the sequence dimension (all tokens)
        pooled_output = self.pooling(x) # shape --> [batch_size, embed_size]

        # Final Linear Classification Layer
        return self.Linear(pooled_output) # shape --> [batch_size, n_classes] or [batch_size, 1]

# Example usage from your code:
if __name__ == "__main__":
    print("Testing the VisionTransformer model...")
    # Create a dummy video tensor: batch size 1, 16 frames, 224x224 spatial, 3 channels (RGB)
    rnd_vid = tf.random.uniform(shape=[1, 16, 224, 224, 3], dtype=tf.float32)
    print(f"Input video shape: {rnd_vid.shape}")

    # Instantiate the model
    model = VisionTransformer(
        n_heads=8,
        n_classes=2, # Binary classification example
        img_size=224,
        mlp_dropout=0.1,
        pos_dropout=0.0,
        attn_dropout=0.0,
        embed_size=198,
        patch_size=8, # Spatial patch size HxW
        n_blocks=4,
        mlpHidden_size=198 * 4, # Standard practice: 4 * embed_size
    )

    # Build the model explicitly (optional, call() does it implicitly)
    # model.build(rnd_vid.shape)
    # print(f"Number of patches (inferred after build): {model.proj.n_patches}")
    # print(f"Positional embedding shape: {model.pos_embed.shape}")


    # Get the model output
    output = model(rnd_vid)
    print(f"Output shape: {output.shape}") # Expected shape: [1, 1] for binary classification
    print("Model output (logits/probabilities):")
    print(output)

    # You would then compile and train the model:
    # model.compile(optimizer='adam',
    #              loss=tf.keras.losses.BinaryCrossentropy() if model.Linear.units == 1 else tf.keras.losses.CategoricalCrossentropy(),
    #              metrics=['accuracy'])
    # model.fit(...)
    print("\nModel test successful!")
```

**Explanation:**

* The `__init__` method sets up all the necessary layers: `Projection`, learnable `cls_token` and `pos_embed` (initialized, but shaped in `build`), `Dropout`, the `Sequential` stack of `TransformerEncoder` blocks, `LayerNormalization`, `GlobalAvgPool1D`, and the final `Dense` classification layer.
* The `build` method is crucial here. It's called with the input shape and is where layers that depend on the input shape (like `self.pos_embed`, which depends on the number of patches calculated by `self.proj.build`) are properly initialized.
* The `call` method defines the forward pass, chaining the layers in the order of the architecture flow: Project -> Concat CLS -> Add Pos Embed -> Dropout -> Encoder Blocks -> Layer Norm -> Pool -> Linear.
* The `if __name__ == "__main__":` block demonstrates how to instantiate the model with specific parameters and pass a dummy video tensor through it to check the output shape. It also shows how you might compile and train it (commented out).

### Foundational Concepts Recap

This Vision Transformer for video relies on several key deep learning concepts:

* **Transformers:** Leveraging the self-attention mechanism to model long-range dependencies within the sequence of video tubelets.
* **Self-Attention:** A mechanism that allows each element in a sequence to weigh the importance of all other elements, creating a context-aware representation.
* **Positional Encoding:** Injecting information about the position of each element in the sequence, as transformers are inherently order-agnostic.
* **Residual Connections:** Adding the input of a sub-layer to its output to facilitate training of deep networks by preventing vanishing gradients.
* **Layer Normalization:** Normalizing activations across features to stabilize training.
* **Tubelet Embedding:** Adapting image patching to video by creating spatio-temporal patches using 3D convolutions.

### References

This architecture is inspired by and builds upon foundational work in Transformers and Vision Transformers:

* **Attention Is All You Need:** The original paper introducing the Transformer architecture.
    * Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez Gomez, N., ... & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems*, 30.
* **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale:** The paper that introduced the Vision Transformer (ViT) for image classification.
    * Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ... & Houlsby, N. (2020). An image is worth 16x16 words: Transformers for image recognition at scale. *arXiv preprint arXiv:2010.11929*.
* **ViViT: A Video Vision Transformer:** A prominent work exploring different ways to apply Vision Transformers to video, including the concept of spatio-temporal attention and variations of patching/pooling. Your "Tubelet Embedding" is related to the "Spatio-temporal Attention" method discussed here.
    * Arnab, A., Dehghani, M., Heigold, G., Wang, C., Wu, J., & Schmid, C. (2021). Vivit: A video vision transformer. *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pp. 6836-6846.

### Conclusion

You've successfully implemented a custom Vision Transformer architecture adapted for video classification in TensorFlow! By using `Conv3D` for spatio-temporal patching (Tubelet Embedding), incorporating positional embeddings, and leveraging the powerful Transformer encoder blocks, this model can learn complex spatio-temporal relationships in video data.

While this implementation provides a solid foundation, exploring variations in tubelet size, stride, different positional encoding schemes, and more advanced attention mechanisms (like those discussed in ViViT) could further enhance performance for various video understanding tasks.

Feel free to experiment with the parameters, train it on a video dataset, and explore its capabilities! Check out the full code in the GitHub repository linked at the beginning of this post.