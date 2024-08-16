class CorrelationAccuracyMetric(tf.keras.metrics.Metric):
    def __init__(self, name='correlation_accuracy', **kwargs):
        super(CorrelationAccuracyMetric, self).__init__(name=name, **kwargs)
        self.total_corr = self.add_weight(name='total_corr', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(y_true, [-1, 1])
        mask = tf.equal(y_true, tf.transpose(y_true))
        mask = tf.cast(mask, dtype=tf.float32)
        similarity = tf.matmul(y_pred, y_pred, transpose_b=True)
        
        # Zero out the diagonal (self-similarity)
        similarity -= tf.eye(tf.shape(similarity)[0])

        # Apply the mask
        masked_similarity = similarity * mask

        # Extract upper triangle elements
        upper_triangle_indices = tf.linalg.band_part(masked_similarity, 0, -1) - tf.linalg.band_part(masked_similarity, 0, 0)
        upper_triangle_values = tf.boolean_mask(masked_similarity, upper_triangle_indices)
        
        # Filter out non-zero elements
        non_zero_upper_triangle_values = tf.boolean_mask(upper_triangle_values, tf.not_equal(upper_triangle_values, 0))

        # Calculate the mean of the non-zero upper triangle elements
        corr_metric = tf.reduce_mean(non_zero_upper_triangle_values)
        
        self.total_corr.assign_add(corr_metric)
        self.count.assign_add(1.0)  # Increment by 1.0 for each batch update

    def result(self):
        return self.total_corr / self.count

    def reset_states(self):
        self.total_corr.assign(0)
        self.count.assign(0)
