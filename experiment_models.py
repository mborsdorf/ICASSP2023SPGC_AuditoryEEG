"""Different model architectures."""

import tensorflow as tf

# Multi-Head Attention block
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self,  embed_dim, num_heads, ff_dim):
        super(TransformerBlock, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([tf.keras.layers.Dense(ff_dim, activation="relu"), tf.keras.layers.Dense(embed_dim), ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(0.5)
        self.dropout2 = tf.keras.layers.Dropout(0.5)

    def build(self, input_shape):
        super(TransformerBlock, self).build(input_shape)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)  
        ffn_output = self.dropout2(ffn_output, training=training)
        out = self.layernorm2(out1 + ffn_output)
        return out


#### Baseline model ####
def dilation_model(
    time_window=None,
    eeg_input_dimension=64,
    env_input_dimension=1,
    layers=3,
    kernel_size=3,
    spatial_filters=8,
    dilation_filters=16,
    activation="relu",
    compile=True,
    inputs=tuple(),
):
    """Convolutional dilation model.

    Code was taken and adapted from
    https://github.com/exporl/eeg-matching-eusipco2020

    Parameters
    ----------
    time_window : int or None
        Segment length. If None, the model will accept every time window input
        length.
    eeg_input_dimension : int
        number of channels of the EEG
    env_input_dimension : int
        dimemsion of the stimulus representation.
        if stimulus == envelope, env_input_dimension =1
        if stimulus == mel, env_input_dimension =28
    layers : int
        Depth of the network/Number of layers
    kernel_size : int
        Size of the kernel for the dilation convolutions
    spatial_filters : int
        Number of parallel filters to use in the spatial layer
    dilation_filters : int
        Number of parallel filters to use in the dilation layers
    activation : str or list or tuple
        Name of the non-linearity to apply after the dilation layers
        or list/tuple of different non-linearities
    compile : bool
        If model should be compiled
    inputs : tuple
        Alternative inputs

    Returns
    -------
    tf.Model
        The dilation model


    References
    ----------
    Accou, B., Jalilpour Monesi, M., Montoya, J., Van hamme, H. & Francart, T.
    Modeling the relationship between acoustic stimulus and EEG with a dilated
    convolutional neural network. In 2020 28th European Signal Processing
    Conference (EUSIPCO), 1175–1179, DOI: 10.23919/Eusipco47968.2020.9287417
    (2021). ISSN: 2076-1465.

    Accou, B., Monesi, M. J., hamme, H. V. & Francart, T.
    Predicting speech intelligibility from EEG in a non-linear classification
    paradigm. J. Neural Eng. 18, 066008, DOI: 10.1088/1741-2552/ac33e9 (2021).
    Publisher: IOP Publishing
    """
    # If different inputs are required
    if len(inputs) == 3:
        eeg, env1, env2 = inputs[0], inputs[1], inputs[2]
    else:
        eeg = tf.keras.layers.Input(shape=[time_window, eeg_input_dimension])
        env1 = tf.keras.layers.Input(shape=[time_window, env_input_dimension])
        env2 = tf.keras.layers.Input(shape=[time_window, env_input_dimension])

    # Activations to apply
    if isinstance(activation, str):
        activations = [activation] * layers
    else:
        activations = activation

    env_proj_1 = env1
    env_proj_2 = env2
    # Spatial convolution
    eeg_proj_1 = tf.keras.layers.Conv1D(spatial_filters, kernel_size=1)(eeg)

    # Construct dilation layers
    for layer_index in range(layers):
        # dilation on EEG
        eeg_proj_1 = tf.keras.layers.Conv1D(
            dilation_filters,
            kernel_size=kernel_size,
            dilation_rate=kernel_size**layer_index,
            strides=1,
            activation=activations[layer_index],
        )(eeg_proj_1)

        # Dilation on envelope data, share weights
        env_proj_layer = tf.keras.layers.Conv1D(
            dilation_filters,
            kernel_size=kernel_size,
            dilation_rate=kernel_size**layer_index,
            strides=1,
            activation=activations[layer_index],
        )
        env_proj_1 = env_proj_layer(env_proj_1)
        env_proj_2 = env_proj_layer(env_proj_2)

    # Comparison
    cos1 = tf.keras.layers.Dot(1, normalize=True)([eeg_proj_1, env_proj_1])
    cos2 = tf.keras.layers.Dot(1, normalize=True)([eeg_proj_1, env_proj_2])

    # Classification
    out = tf.keras.layers.Dense(1, activation="sigmoid")(
        tf.keras.layers.Flatten()(tf.keras.layers.Concatenate()([cos1, cos2]))
    )

    model = tf.keras.Model(inputs=[eeg, env1, env2], outputs=[out])

    if compile:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            metrics=["acc"],
            loss=["binary_crossentropy"],
        )
        print(model.summary())
    return model


#### Model: MHA+DC for EEG and DC for speech stimulus ####
def eeg_mha_dc_speech_dc_model(
    time_window=None,
    eeg_input_dimension=64,
    env_input_dimension=1,
    layers=3,
    kernel_size=3,
    dilation_filters=16,
    activation="relu",
    compile=True,
    inputs=tuple(),
):
    """Convolutional dilation model.

        Code was taken and adapted from
        https://github.com/exporl/eeg-matching-eusipco2020

        Parameters
        ----------
        time_window : int or None
            Segment length. If None, the model will accept every time window input
            length.
        eeg_input_dimension : int
            number of channels of the EEG
        env_input_dimension : int
            dimemsion of the stimulus representation.
            if stimulus == envelope, env_input_dimension =1
            if stimulus == mel, env_input_dimension =28
        layers : int
            Depth of the network/Number of layers
        kernel_size : int
            Size of the kernel for the dilation convolutions
        dilation_filters : int
            Number of parallel filters to use in the dilation layers
        activation : str or list or tuple
            Name of the non-linearity to apply after the dilation layers
            or list/tuple of different non-linearities
        compile : bool
            If model should be compiled
        inputs : tuple
            Alternative inputs

        Returns
        -------
        tf.Model
            The dilation model


        References
        ----------
        Accou, B., Jalilpour Monesi, M., Montoya, J., Van hamme, H. & Francart, T.
        Modeling the relationship between acoustic stimulus and EEG with a dilated
        convolutional neural network. In 2020 28th European Signal Processing
        Conference (EUSIPCO), 1175–1179, DOI: 10.23919/Eusipco47968.2020.9287417
        (2021). ISSN: 2076-1465.

        Accou, B., Monesi, M. J., hamme, H. V. & Francart, T.
        Predicting speech intelligibility from EEG in a non-linear classification
        paradigm. J. Neural Eng. 18, 066008, DOI: 10.1088/1741-2552/ac33e9 (2021).
        Publisher: IOP Publishing
        """
    # If different inputs are required
    if len(inputs) == 3:
        eeg, env1, env2 = inputs[0], inputs[1], inputs[2]
    else:
        eeg = tf.keras.layers.Input(shape=[time_window, eeg_input_dimension])
        env1 = tf.keras.layers.Input(shape=[time_window, env_input_dimension])
        env2 = tf.keras.layers.Input(shape=[time_window, env_input_dimension])

    # Activations to apply
    if isinstance(activation, str):
        activations = [activation] * layers
    else:
        activations = activation

    env_proj_1 = env1
    env_proj_2 = env2

    # Multi-Head Attention
    transformer_block_1 = TransformerBlock(embed_dim=eeg_input_dimension, num_heads=2, ff_dim=32)
    eeg_proj_1 = transformer_block_1(eeg)

     # Construct dilation layers
    for layer_index in range(layers):
        # dilation on EEG
        eeg_proj_1 = tf.keras.layers.Conv1D(
            dilation_filters,
            kernel_size=kernel_size,
            dilation_rate=kernel_size**layer_index,
            strides=1,
            activation=activations[layer_index],
        )(eeg_proj_1)

        # Dilation on envelope data, share weights
        env_proj_layer = tf.keras.layers.Conv1D(
            dilation_filters,
            kernel_size=kernel_size,
            dilation_rate=kernel_size**layer_index,
            strides=1,
            activation=activations[layer_index],
        )
        env_proj_1 = env_proj_layer(env_proj_1)
        env_proj_2 = env_proj_layer(env_proj_2)

    # Comparison
    cos1 = tf.keras.layers.Dot(1, normalize=True)([eeg_proj_1, env_proj_1])
    cos2 = tf.keras.layers.Dot(1, normalize=True)([eeg_proj_1, env_proj_2])

    # Classification
    out = tf.keras.layers.Dense(1, activation="sigmoid")(
        tf.keras.layers.Flatten()(tf.keras.layers.Concatenate()([cos1, cos2]))
    )

    model = tf.keras.Model(inputs=[eeg, env1, env2], outputs=[out])

    if compile:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            metrics=["acc"],
            loss=["binary_crossentropy"],
        )
        print(model.summary())
    return model


#### Model: MHA+DC for EEG and GRU+DC for speech stimulus ####
def eeg_mha_dc_speech_gru_dc_model(
    time_window=None,
    eeg_input_dimension=64,
    env_input_dimension=1,
    layers=3,
    kernel_size=3,
    dilation_filters=16,
    activation="relu",
    compile=True,
    inputs=tuple(),
):
    """Convolutional dilation model.

    Code was taken and adapted from
    https://github.com/exporl/eeg-matching-eusipco2020

    Parameters
    ----------
    time_window : int or None
        Segment length. If None, the model will accept every time window input
        length.
    eeg_input_dimension : int
        number of channels of the EEG
    env_input_dimension : int
        dimemsion of the stimulus representation.
        if stimulus == envelope, env_input_dimension =1
        if stimulus == mel, env_input_dimension =28
    layers : int
        Depth of the network/Number of layers
    kernel_size : int
        Size of the kernel for the dilation convolutions
    dilation_filters : int
        Number of parallel filters to use in the dilation layers
    activation : str or list or tuple
        Name of the non-linearity to apply after the dilation layers
        or list/tuple of different non-linearities
    compile : bool
        If model should be compiled
    inputs : tuple
        Alternative inputs

    Returns
    -------
    tf.Model
        The dilation model


    References
    ----------
    Accou, B., Jalilpour Monesi, M., Montoya, J., Van hamme, H. & Francart, T.
    Modeling the relationship between acoustic stimulus and EEG with a dilated
    convolutional neural network. In 2020 28th European Signal Processing
    Conference (EUSIPCO), 1175–1179, DOI: 10.23919/Eusipco47968.2020.9287417
    (2021). ISSN: 2076-1465.

    Accou, B., Monesi, M. J., hamme, H. V. & Francart, T.
    Predicting speech intelligibility from EEG in a non-linear classification
    paradigm. J. Neural Eng. 18, 066008, DOI: 10.1088/1741-2552/ac33e9 (2021).
    Publisher: IOP Publishing
    """
    # If different inputs are required
    if len(inputs) == 3:
        eeg, env1, env2 = inputs[0], inputs[1], inputs[2]
    else:
        eeg = tf.keras.layers.Input(shape=[time_window, eeg_input_dimension])
        env1 = tf.keras.layers.Input(shape=[time_window, env_input_dimension])
        env2 = tf.keras.layers.Input(shape=[time_window, env_input_dimension])

    # Activations to apply
    if isinstance(activation, str):
        activations = [activation] * layers
    else:
        activations = activation

    # Multi-Head Attention
    transformer_block_1 = TransformerBlock(embed_dim=eeg_input_dimension, num_heads=2, ff_dim=32)
    eeg_proj_1 = transformer_block_1(eeg)

    # Gated Recurrent Unit
    gru_model = tf.keras.layers.GRU(env_input_dimension, return_sequences=True)
    env_proj_1 = gru_model(env1)
    env_proj_2 = gru_model(env2)

    # Construct dilation layers
    for layer_index in range(layers):
        # dilation on EEG
        eeg_proj_1 = tf.keras.layers.Conv1D(
            dilation_filters,
            kernel_size=kernel_size,
            dilation_rate=kernel_size**layer_index,
            strides=1,
            activation=activations[layer_index],
        )(eeg_proj_1)

        # Dilation on envelope data, share weights
        env_proj_layer = tf.keras.layers.Conv1D(
            dilation_filters,
            kernel_size=kernel_size,
            dilation_rate=kernel_size**layer_index,
            strides=1,
            activation=activations[layer_index],
        )
        env_proj_1 = env_proj_layer(env_proj_1)
        env_proj_2 = env_proj_layer(env_proj_2)

    # Comparison
    cos1 = tf.keras.layers.Dot(1, normalize=True)([eeg_proj_1, env_proj_1])
    cos2 = tf.keras.layers.Dot(1, normalize=True)([eeg_proj_1, env_proj_2])

    # Classification
    out = tf.keras.layers.Dense(1, activation="sigmoid")(
        tf.keras.layers.Flatten()(tf.keras.layers.Concatenate()([cos1, cos2]))
    )

    model = tf.keras.Model(inputs=[eeg, env1, env2], outputs=[out])

    if compile:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            metrics=["acc"],
            loss=["binary_crossentropy"],
        )
        print(model.summary())
    return model
