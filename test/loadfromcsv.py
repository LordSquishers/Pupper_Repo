# import numpy
#
# filename = '18'
#
# data = numpy.loadtxt(open(filename + ".csv", "rb"), delimiter=",")
# print(data.shape)
# # print(data)
#
# xs = None
# ys = None
# frames = None
#
# xs = data[0, :]
# ys = data[1, :]
# frames = data[2, :]
#
# transposed_data = numpy.zeros((data.shape[1], data.shape[0]))
# row_idx = 0
# f = open("trnsp_" + filename + ".txt", "a")
# for row in transposed_data:
#     dataline = str(frames[row_idx]) + ' ' + filename + ' ' + str(xs[row_idx]) + ' ' + str(ys[row_idx]) + '\n'
#     f.write(dataline)
#     row_idx += 1
#
# f.close()
