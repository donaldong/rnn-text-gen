import tensorflow
class RNN_Text_Gen:
        def __init__(self, input_len, unique_chars_count):
                #rnn cell
                self.neurons_per_cell = 100
                self.RNN_Cell = tensorflow.nn.rnn_cell.BasicRNNCell(self.neurons_per_cell)
                #dynamic RNN
                self.output_place = tensorflow.placeholder(tensorflow.float32, shape=(None, input_len, unique_chars_count))
                self.input_place = tensorflow.placeholder(tensorflow.float32, shape=(None, input_len,1))
                self.Dynamic_rnn, self.state = tensorflow.nn.dynamic_rnn(self.RNN_Cell, self.input_place, dtype=tensorflow.float32)
                #Training Vars
                print(
                "\n----------------------------------------------------------------\n",
                self.Dynamic_rnn,
                "\n----------------------------------------------------------------\n")
                
                cost = tensorflow.reduce_mean(
                        tensorflow.nn.softmax_cross_entropy_with_logits(
                                logits=self.Dynamic_rnn, 
                                labels=self.output_place
                        )
                )
                self.optimizer = tensorflow.train.AdamOptimizer().minimize(cost)
                self.loss = None #calculated later
        def predict():
                pass
        def score():
                pass
        def fit():
                pass
                
	        
