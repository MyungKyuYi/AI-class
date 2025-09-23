# -------------------------------------------------
# 5) 모델 정의 (작고 탄탄한 MLP)
# -------------------------------------------------
inputs = tf.keras.Input(shape=(X_train.shape[1],))
x = tf.keras.layers.Dense(64)(inputs)
x = tf.keras.layers.ReLU()(x)

x = tf.keras.layers.Dense(32)(x)
x = tf.keras.layers.ReLU()(x)

outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=accuracy)

model.summary()

# -------------------------------------------------
# 7) 학습
# -------------------------------------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=200,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=2,
)
