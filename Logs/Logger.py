import datetime

import Config


class Logger:
    """
    Class used to create log files
    """
    def __init__(self, file_name, logger_type):
        self.file_name = file_name
        f = open(file_name, "w")
        if logger_type == Config.CONSENSUS_LOGGER:
            f.write("time,received_weight,sending_agent,weight\n")
        elif logger_type == Config.MESSAGE_LOGGER:
            f.write("time,send_or_recv,id,communicating_agent\n")
        elif logger_type == Config.TRAINING_LOGGER:
            f.write("time,training_accuracy,training_loss,test_accuracy,test_loss\n")
        elif logger_type == Config.WEIGHT_LOGGER:
            f.write("time,train_or_consensus,first_layer_weight,first_layer_bias,second_layer_weight,"
                    "second_layer_bias\n")
        elif logger_type == Config.EPSILON_LOGGER:
            f.write("time,value\n")
        elif logger_type == Config.TRAINING_TIME_LOGGER:
            f.write("time,start_or_stop\n")
        f.close()

    def write_to_file(self, content):
        """
        Writes a line of content in a log file
        :param content:
        :return:
        """
        file = open(self.file_name, "a")
        file.write(str(datetime.datetime.now()) + "," + content + "\n")
        file.close()
