def supervised_contrastive_loss(labels, projections, temperature=0.1):
    labels = tf.reshape(labels, [-1, 1])
    mask = tf.equal(labels, tf.transpose(labels))

    dot_product = tf.matmul(projections, tf.transpose(projections))  / temperature    

    log_prob = dot_product - tf.reduce_logsumexp(dot_product, axis=1, keepdims=True)
    mask = tf.cast(mask, dtype=tf.float32)
    positive_log_prob = tf.reduce_sum(log_prob * mask, axis=1) / tf.reduce_sum(mask, axis=1)
    
    loss = -tf.reduce_mean(positive_log_prob)
    return loss
