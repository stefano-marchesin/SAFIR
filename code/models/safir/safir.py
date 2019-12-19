import tensorflow as tf


class SAFIR(object):
    """build the graph of SAFIR representation learning component"""

    def __init__(self, _word_vocab_size, _corpus_size, _synsets, _options, _concept_vocab_size=None):
        self.word_vocab_size = _word_vocab_size
        self.corpus_size = _corpus_size
        self.synsets = _synsets
        self.options = _options
        if self.options.poly:
            self.concept_vocab_size = _concept_vocab_size

        """PARAMETER INITIALIZATION"""
        opts = self.options

        with tf.name_scope('embeddings'):
            self.word_embs = tf.get_variable('word_embs', shape=[self.word_vocab_size, opts.word_size],
                                             initializer=tf.glorot_uniform_initializer(seed=opts.seed), trainable=True)

            if opts.poly:
                self.concept_embs = tf.get_variable('concept_embs', shape=[self.concept_vocab_size, opts.concept_size],
                                                initializer=tf.glorot_uniform_initializer(seed=opts.seed), trainable=True)
                self.row_ixs = [[0, i] for i in range(opts.concept_size)]
                self.masked_concept_embs = tf.scatter_nd_update(self.concept_embs, self.row_ixs, tf.zeros([opts.concept_size,]))

            self.doc_embs = tf.get_variable('doc_embs', shape=[self.corpus_size, opts.doc_size],
                                            initializer=tf.glorot_uniform_initializer(seed=opts.seed), trainable=True)

        with tf.name_scope('weights'):
            self.proj_weights = tf.get_variable('proj_weights', shape=[opts.doc_size, opts.word_size],
                                                initializer=tf.glorot_uniform_initializer(seed=opts.seed), trainable=True)

        """PLACEHOLDERS"""
        with tf.name_scope('placeholders'):
            self.ngram_words = tf.placeholder(tf.int32, shape=[opts.batch_size, opts.ngram_size])
            if opts.poly:
                self.ngram_concepts = tf.placeholder(tf.int32, shape=[opts.batch_size, opts.ngram_size])
            self.labels = tf.placeholder(tf.int32, shape=[opts.batch_size])
            self.negative_labels = tf.placeholder(tf.int32, shape=[opts.batch_size, opts.neg_samples])

        """EMBEDDING LOOKUPS"""
        with tf.name_scope('lookups'):
            # embedding lookups
            self.words = tf.nn.embedding_lookup(self.word_embs, self.ngram_words)
            if opts.poly:
                with tf.control_dependencies([self.masked_concept_embs]):
                    self.concepts = tf.nn.embedding_lookup(self.masked_concept_embs, self.ngram_concepts)
            else:
                self.concepts = None
            self.syn_words = tf.nn.embedding_lookup(self.word_embs, self.synsets[:, :, 0])
            self.true_docs = tf.nn.embedding_lookup(self.doc_embs, self.labels)
            self.negative_docs = tf.nn.embedding_lookup(self.doc_embs, self.negative_labels)

        """FORWARD PASS"""
        with tf.name_scope('forward_pass'):
            self.proj_ngrams = self.ngrams2docs(self.words, self.concepts, opts.l2_norm)
            self.stand_ngrams = self.standardize_batch(self.proj_ngrams)
            # true logits [batch_size]
            self.true_logits = self.compute_true_logits(self.stand_ngrams, self.true_docs)
            # negative logits [batch_size, neg_samples]
            self.neg_logits = self.compute_negative_logits(self.stand_ngrams, self.negative_docs)
            # semantic logits [synsets_size]
            if opts.sem_term > 0.0:
                self.sem_logits = self.compute_semantic_logits(self.syn_words) 

        """LOSS OPERATION"""
        with tf.name_scope('loss_ops'):
            # nce loss
            self.text_loss = self.text_matching_loss()
            # regularization loss
            self.reg_loss = self.regularization_loss()
            # total loss
            self.loss = self.text_loss + tf.constant(opts.reg_term) * self.reg_loss
            # semantic loss
            if opts.sem_term > 0.0:
                self.sem_loss = self.semantic_matching_loss()
                # update total loss
                self.loss += tf.constant(opts.sem_term) * self.sem_loss 
            tf.summary.scalar('loss', self.loss)

        """OPTIMIZATION OPERATION"""
        with tf.name_scope('opt_ops'):
            optimizer = tf.train.AdamOptimizer()
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                # ensures we execute the update_ops before performing the train_step
                self.train_op = optimizer.minimize(self.loss)

    def sum(self, words, concepts):
        """sum words and concepts element-wise to get senses"""
        with tf.name_scope('sum_op'):
            if concepts is None:
                return words
            else:
                return tf.add(words, concepts)

    def average(self, ngrams):
        """average ngram inputs: [batch_size, emb_size]"""
        with tf.name_scope('average'):
            return tf.reduce_mean(ngrams, axis=1)

    def norm(self, ngrams):
        """l2 normalize ngrams"""
        with tf.name_scope('norm'):
            return tf.nn.l2_normalize(ngrams, axis=1)

    def projection(self, ngrams):
        """project ngrams from word to doc embeddings space"""
        with tf.name_scope('projection'):
            return tf.transpose(tf.matmul(self.proj_weights, ngrams, transpose_b=True))

    def ngrams2docs(self, words, concepts, norm=True):
        """composition function: projection(norm(average(sum(words, concepts))))"""
        with tf.name_scope('ngram2doc'):
            if norm:
                return self.projection(self.norm(self.average(self.sum(words, concepts))))
            else:
                return self.projection(self.average(self.sum(words, concepts)))

    def standardize_batch(self, batch):
        """standardization operation to reduce internal covariate shift + hard tanh"""
        with tf.name_scope('standardization'):
            batch_norm = tf.layers.batch_normalization(batch, axis=1, scale=False, training=True)
            return tf.clip_by_value(batch_norm, clip_value_min=-1.0, clip_value_max=1.0)

    def compute_true_logits(self, ngrams, true_docs):
        """compute true logits"""
        with tf.name_scope('true_logits'):
            return tf.reduce_sum(tf.multiply(true_docs, ngrams), axis=1)

    def compute_negative_logits(self, ngrams, negative_docs):
        """compute negative logits"""
        with tf.name_scope('negative_logits'):
            return tf.matmul(negative_docs, ngrams[..., None])[..., 0]  # add and remove extra dimension

    def compute_semantic_logits(self, syn_words): 
        """compute semantic logits"""
        with tf.name_scope('semantic_logits'):
            return tf.reduce_sum(tf.multiply(syn_words[:, 0], syn_words[:, 1]), axis=1)

    def text_matching_loss(self):
        """compute nce loss"""
        with tf.name_scope('nce_loss'):
            true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(self.true_logits),
                logits=self.true_logits)
            neg_xent = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(self.neg_logits),
                logits=self.neg_logits), axis=1)
            # compute nce loss with scaled negative examples
            nce_loss = tf.reduce_mean(((self.options.neg_samples + 1.0) / (2.0 * self.options.neg_samples)) *
                                      (self.options.neg_samples * true_xent + neg_xent))
            tf.summary.scalar('nce_loss', nce_loss)
            return nce_loss

    def semantic_matching_loss(self): 
        """compute semantic matching loss"""
        with tf.name_scope('semantic_loss'):
            xent = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(self.sem_logits),
                logits=self.sem_logits)
            # compute semantic loss
            sem_loss = tf.reduce_sum(xent) / self.options.batch_size
            tf.summary.scalar('sem_loss', sem_loss)
            return sem_loss

    def regularization_loss(self):
        """compute regularization loss"""
        with tf.name_scope('regularization_loss'):
            reg_loss = tf.nn.l2_loss(self.word_embs) + tf.nn.l2_loss(self.proj_weights) + tf.nn.l2_loss(self.doc_embs)
            if self.options.poly:
                reg_loss += tf.nn.l2_loss(self.concept_embs)
            reg_loss /= self.options.batch_size
            tf.summary.scalar('reg_loss', reg_loss)
            return reg_loss
