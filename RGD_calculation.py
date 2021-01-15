

from CNN_model import predict_image
import cv2
import PIL, os
import pandas as pd


def get_dimentions(extracted, depths_list, step):
    '''
    :param extracted: core tray without the background
    :param depths_list: all depths in the image
    :param step: a defined step for the scanning window
    :return: no_rows in the tray, no_columns depending on the step, the size of the scanning window
    '''
    # the number of rows from the number of depths
    no_rows= len(depths_list ) -1
    # the final cropped image size equals to the height of the row
    image_size = int(extracted.shape[0] /no_rows)
    no_columns = int((extracted.shape[1 ] -image_size )/ step)                                                                 # decide a step to move and crop along each row
    return(no_rows, no_columns, image_size)



def get_rows(extracted, no_rows, image_size):
    # extract each row and create a list contains the images of all rows
    rows =[]
    for y in range(0, extracted.shape[0], image_size):
        if len(rows) == no_rows:
            break
        row = extracted[y:y + image_size, 0:extracted.shape[1]]
        rows.append(row)
    return (rows)


def draw_rec(overlay2, x1, x2, y1, y2, color, r1, r2):
    #optional: for plotting and visualization
    x1 += r1
    x2 += r1
    y1 += r2
    y2 += r2
    hight = y2 - y1
    y1 += int(0.1 * hight)
    y2 -= int(0.1 * hight)
    overlay2 = cv2.rectangle(overlay2, (x1, y1), (x2, y2), color, -1)
    return (overlay2)




def check_length(Li, threshold, Lt):
    if Li >= threshold:
        Lt += Li


# take each row from the previous list, crop the images and calculate the depth of each image
'''def final(extracted, depths_list, step, pixels_eq_10cm, model, class_to_idx, img, r1, r2):
    #the main function that uses all the obtained parameters and output RQD for each row of the processed image
    no_rows, no_columns, image_size = get_dimentions(extracted, depths_list, step)
    rows = get_rows(extracted, no_rows,image_size)
    image_names = []
    depths_from = []
    depths_to = []
    RQDs = []

    # define a threshold of 10 cm
    threshold = pixels_eq_10cm

    # optional:for plotting
    # output2 = img.copy()
    # overlay2 = img.copy()
    # alpha = 0.2

    Lt_image = 0

    for i, row in enumerate(rows):
        Lt = 0
        Li = 0
        k = 0
        print('row {} of {}'.format((i + 1), no_rows))  #  to monitor the progress

        depth_from = float(depths_list[i])
        depth_to = float(depths_list[i + 1])
        depths_from.append(depth_from)
        depths_to.append(depth_to)
        row_length = depth_to - depth_from
        class_names_row = []

        j = 0
        for x in range(0, row.shape[1], step):
            if j == no_columns:
                break
            final_image = row[0:row.shape[0], x:x + image_size]
            #use the CNN model to predict
            class_name = predict_image(final_image,model, class_to_idx)
            class_names_row.append(class_name)
            if class_name == 'whole':
                if j == 0 or class_names_row[j - 1] != 'whole':
                    k += 1
                    Li = image_size
                    x1 = x
                    y1 = i * image_size  # this is constant maybe fix it outside
                    x2 = x + image_size
                    y2 = i * image_size + image_size
                else:  # when class_names_row[j-1] == 'whole'
                    Li += step
                    x2 += step

                if j == no_columns:
                    if Li >= threshold:  # if an intact core at the end of the row >10cm
                        Lt += Li
                    #     overlay2 = draw_rec(overlay2, x1, x2, y1, y2, (0, 255, 0), r1, r2)
                    # else:  # if an intact core at the end of the row <10cm
                    #     overlay2 = draw_rec(overlay2, x1, x2, y1, y2, (0, 0, 255), r1, r2)

            else:  # when class_names_row[j] != 'whole'
                if j > 0 and class_names_row[j - 1] == 'whole':
                    # print('intact:',k, 'length', Li)
                    if Li >= threshold:
                         Lt += Li
                    #     overlay2 = draw_rec(overlay2, x1, x2, y1, y2, (0, 255, 0), r1, r2)
                    # else:
                    #     overlay2 = draw_rec(overlay2, x1, x2, y1, y2, (0, 0, 255), r1, r2)
            j += 1
        Lt_m = Lt / (threshold * 10)  # convert Lt from pixels to m (threshold is 10 cm )
        RQD = 100 * Lt_m / row_length
        RQDs.append(RQD)
        Lt_image += Lt_m

    # whole image RQD calculation
    depth_from = float(depths_list[0])
    depth_to = float(depths_list[len(rows)])
    depths_from.append(depth_from)
    depths_to.append(depth_to)
    image_length = depth_to - depth_from
    RQD_image = 100 * Lt_image / image_length
    RQDs.append(RQD_image)
    # overlay all rectangles on the original image
    #rec = cv2.addWeighted(overlay2, alpha, output2, 1 - alpha, 0)
    return (RQDs, depths_from, depths_to, no_rows)'''


# get the results in a csv file
def output_report(file_csv, rows_names_total, depths_from_total, depths_to_total, RQDs_total):
    report = pd.DataFrame(
        {'row name': rows_names_total, 'depth from (m)': depths_from_total, 'depth to (m)': depths_to_total,
         'RQD %': RQDs_total}, columns=['row name', 'depth from (m)', 'depth to (m)', 'RQD %']).to_csv(file_csv,
                                                                                                       index=False)
    print('please check your output file')
    return (report)


def final(extracted, depths_list, step, pixels_eq_10cm, model, class_to_idx, img, r1, r2):
    #the main function that uses all the obtained parameters and output RQD for each row of the processed image
    no_rows, no_columns, image_size = get_dimentions(extracted, depths_list, step)
    rows = get_rows(extracted, no_rows,image_size)
    image_names = []
    depths_from = []
    depths_to = []
    RQDs = []

    # define a threshold of 10 cm
    threshold = pixels_eq_10cm

    # optional:for plotting
    output2 = img.copy()
    overlay2 = img.copy()
    alpha = 0.2

    Lt_image = 0

    for i, row in enumerate(rows):
        Lt = 0
        Li = 0
        k = 0
        print('row {} of {}'.format((i + 1), no_rows))  #  to monitor the progress

        depth_from = float(depths_list[i])
        depth_to = float(depths_list[i + 1])
        depths_from.append(depth_from)
        depths_to.append(depth_to)
        row_length = depth_to - depth_from
        class_names_row = []

        j = 0
        for x in range(0, row.shape[1], step):
            if j == no_columns:
                break
            final_image = row[0:row.shape[0], x:x + image_size]
            #use the CNN model to predict
            class_name = predict_image(final_image,model, class_to_idx)
            class_names_row.append(class_name)
            if class_name == 'whole':
                if j == 0 or class_names_row[j - 1] != 'whole':
                    k += 1
                    Li = image_size
                    x1 = x
                    y1 = i * image_size  # this is constant maybe fix it outside
                    x2 = x + image_size
                    y2 = i * image_size + image_size
                else:  # when class_names_row[j-1] == 'whole'
                    Li += step
                    x2 += step

                if j == no_columns:
                    if Li >= threshold:  # if an intact core at the end of the row >10cm
                        Lt += Li
                        #overlay2 = draw_rec(overlay2, x1, x2, y1, y2, (0, 255, 0), r1, r2)
                    # if an intact core at the end of the row <10cm
                    #else: overlay2 = draw_rec(overlay2, x1, x2, y1, y2, (0, 0, 255), r1, r2)

            else:  # when class_names_row[j] != 'whole'
                overlay2 = draw_rec(overlay2, x, x + image_size, i * image_size , i * image_size + image_size, (255, 0, 0), r1, r2)
                if j > 0 and class_names_row[j - 1] == 'whole':
                    #xc = int((x1 + x2) / 2)

                    # print('intact:',k, 'length', Li)
                    if Li >= threshold:
                         Lt += Li
                         #overlay2 = draw_rec(overlay2, x1, x2, y1, y2, (0, 255, 0), r1, r2)
                    #else:
                         #overlay2 = draw_rec(overlay2, x1, x2, y1, y2, (0, 0, 255), r1, r2)
            j += 1
        Lt_m = Lt / (threshold * 10)  # convert Lt from pixels to m (threshold is 10 cm )
        RQD = 100 * Lt_m / row_length
        if RQD > 100 : RQD =100
        RQDs.append(RQD)


    #overlay all rectangles on the original image
    rec = cv2.addWeighted(overlay2, alpha, output2, 1 - alpha, 0)
    return (RQDs, depths_from, depths_to, no_rows, rec)