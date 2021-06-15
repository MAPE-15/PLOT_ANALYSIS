
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')


def plot_analysis(dataset, x=None, y=None, categorized=None, histogram=False):
    """
    AXIS DEFINED CAN BE DEFINED IN WHATEVER CASE, LOWER OR UPPER DON'T MATTER (like Pascal)

    dataset --> give pandas dataframe
    x --> define for x axis, can be more than one, but column names must be defined in the list
    y --> define for y axis, only one, in the list
    categorized --> column name which has categorized values like --> True/False to 1/0, only one, in the list
    histogram --> if want to make histogram plot of all calumns defined

    """

    # when no IVs defined, when x = None
    class Zero_IVs(Exception):
        pass

    # when there is more than one column name in y or categorized
    class More_Than_One_Error(Exception):
        pass

    # if column does not occur in the dataset
    class Column_Not_Present(Exception):
        pass

    # all parameters if defined must be a list
    class Type_List_Error(Exception):
        pass

    # x and y must be defined
    class Must_Be_Defined(Exception):
        pass

    # for histogram parameter, defined as False or True
    class Boolean_Error(Exception):
        pass


    # if x and y are both defined and in a list
    if (x is not None) and (y is not None) and (type(x) == list) and (type(y) == list):

        # make a copy of original (not case modified) column names
        orig_column = list(dataset.columns).copy()

        # set all column names in the dataset to be in lower case
        dataset.columns = [column.lower() for column in list(dataset.columns)]


        # if there is 0 columns for y axis to plot, raise an error
        if len(x) <= 0:
            raise Zero_IVs('!!! Parameter x must have at least one element !!!')

        # if there is more than 1 column for y axis, or none raise an error
        if len(y) != 1:
            raise More_Than_One_Error('!!! Parameter y must have only one element !!!')


        # if categorized is not defined
        if categorized is None:

            # check for every column name if it occurs in the dataset
            for column_name in x + y:
                if column_name.lower() not in list(dataset.columns):
                    raise Column_Not_Present('!!! Column --> ^' + column_name + '^ does not occur in the dataset !!!')


            # make a dict, key --> column name, value --> columns actual elements in numpy array
            df_corr_dict = {}
            for column_name in x + y:
                df_corr_dict[column_name] = np.array(dataset[column_name.lower()])

            # make a pandas DataFrame of them
            df_corr = pd.DataFrame(df_corr_dict)


            # for each x specified
            for x_axis in x:
                # find correlation value between x and y
                corr = np.round(df_corr[[x_axis, ''.join(y)]].corr().iloc[0, 1], 4)

                # size of figure in inches (width, height)
                fig = plt.figure(figsize=(6, 6))
                # make a subplot
                ax1 = plt.subplot2grid((6, 1), (2, 0), rowspan=5, colspan=1, fig=fig)

                # make a scatter plot
                ax1.scatter(dataset[x_axis.lower()], dataset[''.join(y).lower()], marker='D', c='#6f029e')

                # make a text in bbox with corr value
                ax1.text(0.15, 0.72, 'Correlation (r) = ' + str(corr),
                         fontsize=12, transform=plt.gcf().transFigure,
                         bbox={'facecolor': 'blue', 'alpha': 0.2, 'pad': 10})

                # set x label, y label, and suptitle
                ax1.set_xlabel(x_axis)
                ax1.set_ylabel(''.join(y))
                fig.suptitle('\n\n\n' + ''.join(y) + ' (y) VS ' + x_axis + '\n SCATTER PLOT')

                plt.show()


            # lower case all columns defined in x categorized and y, and set in in the list lower_case
            # make a set of those column, because we don't want the same column names multiple times, we want original columns
            lower_case = set([col.lower() for col in x + y])

            # make this matrix plot df with those lower_case column names
            matrix_plot_df = dataset[list(lower_case)]
            print(matrix_plot_df)


            if histogram is False:
                pass

            elif histogram is True:
                # make a histogram of each variable with color blue
                matrix_plot_df.hist(color='blue')
                plt.suptitle('HISTOGRAM PLOT OF EACH VARIABLE')
                plt.show()

            else:
                raise Boolean_Error('!!! Parameter histogram must a boolean value True or False (default) !!!')


            # print out correlation value between each variable
            print('')
            print('CORRELATION MATRIX')
            print(df_corr.corr())

            # !!! pd.plotting.scatter_matrix(dataframe) --> matrix of scatter plots between each variable !!!
            pd.plotting.scatter_matrix(matrix_plot_df, figsize=(9, 9), alpha=0.8, marker='.', color='blue')
            plt.suptitle('SCATTER PLOT BETWEEN EACH VARIABLE\nCORRELATION MATRIX SHOWN IN OUTPUT')
            plt.show()

            # set to the dataset its original column names (not case modified)
            dataset.columns = orig_column



        elif (categorized is not None) and (type(categorized) == list):

            # if there is more than 1 column for categorized, or none raise an error
            if len(categorized) != 1:
                raise More_Than_One_Error('!!! Parameter categorized, if defined, must have only one element !!!')

            for column_name in x + y + categorized:
                if column_name.lower() not in list(dataset.columns):
                    raise Column_Not_Present('!!! Column --> ^' + column_name + '^ does not occur in the dataset !!!')


            # make a dict, key --> column name, value --> columns actual elements in numpy array
            df_corr_dict = {}
            for column_name in x + y:
                df_corr_dict[column_name] = np.array(dataset[column_name.lower()])

            # make a pandas DataFrame of them
            df_corr = pd.DataFrame(df_corr_dict)


            # gotta set the colors for each category (1s and 0s f.e.) for each element
            # matplotlib will know that there is a sequence of numbers and than it will make two colors of them (if there are 2 categories (1s and 0s))
            colors = [i for i in dataset[''.join(categorized).lower()]]

            # for each x specified
            for x_axis in x:
                # find correlation value between x and y
                corr = np.round(df_corr[[x_axis, ''.join(y)]].corr().iloc[0, 1], 4)

                # size of figure in inches (width, height)
                fig = plt.figure(figsize=(6, 6))
                # make a subplot
                ax1 = plt.subplot2grid((6, 1), (2, 0), rowspan=5, colspan=1, fig=fig)

                # make a scatter plot and also set it to a variable, colors of scatter plots will be equal to colors specified few lines above
                scatter = plt.scatter(dataset[x_axis.lower()], dataset[''.join(y).lower()], marker='D', c=colors)

                # make a text in bbox with corr value
                ax1.text(0.15, 0.72, 'Correlation (r) = ' + str(corr),
                         fontsize=12, transform=plt.gcf().transFigure,
                         bbox={'facecolor': 'blue', 'alpha': 0.2, 'pad': 10})

                # make a legend for categorized scatter plot (each scatter category will have its own color, and here we make it to see it in legend, which colors belongs to what category)
                ax1.legend(handles=scatter.legend_elements()[0], labels=list(dataset[''.join(categorized).lower()].unique()))
                ax1.set_xlabel(x_axis)
                ax1.set_ylabel(''.join(y))
                fig.suptitle('\n\n\n' + ''.join(y) + ' (y) VS ' + x_axis + '\n SCATTER PLOT CATEGORIZED')

                plt.show()

            # lower case all columns defined in x categorized and y, and set in in the list lower_case
            # make a set of those column, because we don't want the same column names multiple times, we want original columns
            lower_case = set([col.lower() for col in x + categorized + y])

            # make this matrix plot df with those lower_case column names
            matrix_plot_df = dataset[list(lower_case)]

            if histogram is False:
                pass

            elif histogram is True:
                # make a histogram of each variable with color blue
                matrix_plot_df.hist(color='blue')
                plt.suptitle('HISTOGRAM PLOT OF EACH VARIABLE')
                plt.show()

            else:
                raise Boolean_Error('!!! Parameter histogram must a boolean value True or False (default) !!!')


            # print out correlation value between each variable
            print('')
            print('CORRELATION MATRIX')
            print(df_corr.corr())

            # !!! pd.plotting.scatter_matrix(dataframe) --> matrix of scatter plots between each variable !!!
            pd.plotting.scatter_matrix(matrix_plot_df, figsize=(9, 9), alpha=0.8, marker='.', color='blue')
            plt.suptitle('SCATTER PLOT BETWEEN EACH VARIABLE\nCORRELATION MATRIX SHOWN IN OUTPUT')
            plt.show()

            # set to the dataset its original column names (not case modified)
            dataset.columns = orig_column

        # if categorized is defined bu not in type list, raise an error
        elif (categorized is not None) and (type(categorized) != list):
            raise Type_List_Error('!!! Parameter categorized must be in list type, if defined !!!')


    # both x and y must be defined
    elif (x is None) or (y is None):
        raise Must_Be_Defined('!!! Parameters x and y must be defined !!!')

    # both must be in list type
    elif (type(x) != list) or (type(y) != list):
        raise Type_List_Error('!!! At least one of the parameters is not in type list !!!')
