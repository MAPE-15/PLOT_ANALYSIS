
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style


# x1 = [89, 66, 78, 111, 44, 77, 80, 66, 109, 76]
# x2 = [4, 1, 3, 6, 1, 3, 3, 2, 5, 3]
# x3 = [3.84, 3.19, 3.78, 3.89, 3.57, 3.57, 3.03, 3.51, 3.54, 3.25]
# y1 = [7, 5.4, 6.6, 7.4, 4.8, 6.4, 7, 5.6, 7.3, 6.4]
#
# my_dict = {'x1': np.array(x1), 'x2': np.array(x2), 'x3': np.array(x3), 'y': np.array(y1)}
# df = pd.DataFrame(my_dict)


# to make our plot look a little nicer
style.use('ggplot')

# df --> dataframe made in pandas
# x --> value(s) for x axis, can be multiple columns
# y --> value for y axis, must be only one column
# categorized --> not mandatory to specify, if so, one column max
def plot_corr(df, x=None, y=None, categorized=None, histogram='NO'):


    # CHECK FOR EXCEPTIONS
    try:

        # if there is 0 columns for y axis to plot, raise an error
        if len(x) <= 0:
            raise Exception


        # if there is more than 1 column for y axis, or none raise an error
        if len(y) != 1:
            raise Exception


        if categorized is not None:
            # if any of those column specifies does not occur in the dataframe, raise an error
            for column_name in x + y + categorized:
                if column_name in list(df.columns):
                    pass

                else:
                    raise Exception

            # if there is more than 1 column for categorized, or none raise an error
            if len(categorized) != 1:
                raise Exception

            # make a df of those columns specified for x and y axis and also for categorized, y column will be the last column
            df = df[[x_col for x_col in x] + categorized + y]


        elif categorized is None:
            # if any of those column specifies does not occur in the dataframe, raise an error
            for column_name in x + y:
                if column_name in list(df.columns):
                    pass

                else:
                    raise Exception

            # make a df of those columns specified for x and y axis, y column will be the last column
            df = df[[x_col for x_col in x] + y]


        # make a dict, key --> column name, value --> columns actual elements in numpy array
        df_corr_dict = {}
        for column_name in x + y:
            df_corr_dict[column_name] = np.array(df[column_name])
        # make a pandas DataFrame of them
        df_corr = pd.DataFrame(df_corr_dict)


    except Exception:
        print('')
        print('Oops something went wrong, check if you wrote column names right, or if there is only one y, or only one categorized column !!!')
        make_analysis(df)


    # change just the naming
    all_x = x

    if histogram == 'NO':
        pass

    elif histogram == 'YES':
        # make a histogram of each variable with color blue
        df.hist(color='blue')
        plt.suptitle('HISTOGRAM PLOT OF EACH VARIABLE')
        plt.show()


    # if there is not categorized column
    if categorized is None:

        # for each x specified
        for x_axis in all_x:
            # find correlation value between x and y
            corr = np.round(df_corr[[x_axis, ''.join(y)]].corr().iloc[0, 1], 4)

            # size of figure in inches (width, height)
            fig = plt.figure(figsize=(6, 6))
            # make a subplot
            ax1 = plt.subplot2grid((6, 1), (2, 0), rowspan=5, colspan=1, fig=fig)

            # make a scatter plot
            ax1.scatter(df[x_axis], df[''.join(y)], marker='D', c='#6f029e')

            # make a text in bbox with corr value
            ax1.text(0.15, 0.72, 'Correlation (r) = ' + str(corr),
                     fontsize=12, transform=plt.gcf().transFigure,
                     bbox={'facecolor': 'blue', 'alpha': 0.2, 'pad': 10})

            # set x label, y label, and suptitle
            ax1.set_xlabel(x_axis)
            ax1.set_ylabel(''.join(y))
            fig.suptitle('\n\n\n' + ''.join(y) + ' (y) VS ' + x_axis + '\n SCATTER PLOT')

            plt.show()


        # print out correlation value between each variable
        print('')
        print('CORRELATION MATRIX')
        print(df_corr.corr())

        # !!! pd.plotting.scatter_matrix(dataframe) --> matrix of scatter plots between each variable !!!
        pd.plotting.scatter_matrix(df, figsize=(9, 9), alpha=0.8, marker='.', color='blue')
        plt.suptitle('SCATTER PLOT BETWEEN EACH VARIABLE\nCORRELATION MATRIX SHOWN IN OUTPUT')
        plt.show()


    # if there is a categorized column specified
    elif categorized is not None:

        # gotta set the colors for each category (1s and 0s f.e.) for each element
        # matplotlib will know that there is a sequence of numbers and than it will make two colors of them (if there are 2 categories (1s and 0s))
        colors = [i for i in df[''.join(categorized)]]

        # for each x specified
        for x_axis in all_x:
            # find correlation value between x and y
            corr = np.round(df_corr[[x_axis, ''.join(y)]].corr().iloc[0, 1], 4)

            # size of figure in inches (width, height)
            fig = plt.figure(figsize=(6, 6))
            # make a subplot
            ax1 = plt.subplot2grid((6, 1), (2, 0), rowspan=5, colspan=1, fig=fig)

            # make a scatter plot and also set it to a variable, colors of scatter plots will be equal to colors specified few lines above
            scatter = plt.scatter(df[x_axis], df[''.join(y)], marker='D', c=colors)

            # make a text in bbox with corr value
            ax1.text(0.15, 0.72, 'Correlation (r) = ' + str(corr),
                     fontsize=12, transform=plt.gcf().transFigure,
                     bbox={'facecolor': 'blue', 'alpha': 0.2, 'pad': 10})


            # make a legend for categorized scatter plot (each scatter category will have its own color, and here we make it to see it in legend, which colors belongs to what category)
            ax1.legend(handles=scatter.legend_elements()[0], labels=list(df[''.join(categorized)].unique()))
            ax1.set_xlabel(x_axis)
            ax1.set_ylabel(''.join(y))
            fig.suptitle('\n\n\n' + ''.join(y) + ' (y) VS ' + x_axis + '\n SCATTER PLOT CATEGORIZED')

            plt.show()


        # print out correlation value between each variable
        print('')
        print('CORRELATION MATRIX')
        print(df_corr.corr())

        # !!! pd.plotting.scatter_matrix(dataframe) --> matrix of scatter plots between each variable !!!
        pd.plotting.scatter_matrix(df, figsize=(9, 9), alpha=0.8, marker='.', color='blue')
        plt.suptitle('SCATTER PLOT BETWEEN EACH VARIABLE\nCORRELATION MATRIX SHOWN IN OUTPUT')
        plt.show()



def make_analysis(dataset):

    print('')
    ask_plot = input('Do you wanna make a histogram plot, scatter plot and correlation analysis? Yes/No: ').upper()

    if ask_plot == 'YES':
        print('')
        print('HERE ARE YOUR COLUMN NAMES:', list(dataset.columns))
        print('')

        # ask for x, for column name(s) user wants for x axis, and ask for y, column name user wants for y axis
        ask_x = input('Specify x axis, type column names (in your dataset) you want in your x axis (can be multiple columns): ').split(', ')
        ask_y = input('Specify y axis, type column name (in your dataset) you want in your y axis (MUST be only one column !!!): ').split(', ')

        print('')
        # ask if user wants to specify a categorized column
        ask_categorized = input('Wanna make a categorized scatter plot (if you have in your column f.e. True --> 1, False --> 0)? Yes/No: ').upper()

        # if no, leave it be
        if ask_categorized == 'NO':
            ask_categorized = None

        # if so, type that column which is categorized
        elif ask_categorized == 'YES':
            ask_categorized = input('OK, so type column name which is categorized (MUST be only one column !!!): ').split(', ')

        else:
            print('')
            print('Wrong input Yes/No, try again !!!')
            make_analysis(dataset)

        # make that scatter plot and correlation with those parameters, and set them to scatter_plot_corr arguments
        plot_corr(dataset, x=ask_x, y=ask_y, categorized=ask_categorized, histogram='YES')

        # ask if user wants to try again this scatter plot and correlation analysis
        while True:
            print('')
            ask_again = input('Wanna try scatter plot and correlation analysis again? Yes/No: ').upper()

            if ask_again == 'YES':
                make_analysis(dataset)
                break

            elif ask_again == 'NO':
                print('')
                print('OK, no more scatter plot nor correlation analysis')
                break

            else:
                print('')
                print('Wrong input Yes/No, try again !!!')


    elif ask_plot == 'NO':
        print('')
        print('OK, no scatter plot nor correlation analysis, nor histograms.')

    else:
        print('')
        print('Wrong input Yes/No, try again !!!')
        make_analysis(dataset)

# make_analysis(df)