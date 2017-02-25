#####################################
#                                   #
#                                   #
#   Data Visualization for Python   #
#                                   #
#                                   #
#####################################  

# ------------------------------ Python Data Visualization Basic 
import Matplotlib as Mp # Matplotlib is the most popular Python library for producing plots and other 2D data visualizations








-----------------------------------------------------------------------------------------------------------------------------
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
>                               >  
>                               >
>         matplotlib            >
>                               >
>                               >
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


# plotting (matplotlib > pyplot) - http://matplotlib.org/
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = 8, 6 # Adjust plot size
plt.rcParams.keys() # more choices
# Basic plot
plt.plot(x, y)
# line style maker
plt.plot(x, y, '-') : '-', '--', '-.', ':', '.', '*', ...

# color / legend
plt.plot(x, y, color = 'blue', label='sin(x)')
plt.plot(x, y2, color = 'red', label='cos(x)')
plt.legend(loc=1) # 1:topright, 2:topleft, 3:bottomleft, 4:bottomright

# text / title
plt.title('$sin(x)$ VS $cos(x)$', fontsize = 25)
plt.text(1.6, 0, '$sin(x) > cos(x)$', fontsize = 20) # 1.6, 0 -> x y for coordiates on plot
plt.text(1, -0.5, '$y = cos(x)$', fontsize = 20)
plt.text(3, 0.3, '$y = sin(x)$', fontsize = 20, color = 'blue') # add color


# axis
### axis
plt.xlabel('x', fontsize = 15)
plt.ylabel('y', fontsize = 15)
plt.xticks(fontsize=10,rotation=45);
plt.yticks(fontsize=10,rotation=45);
plt.xlim(-0.5, 5.5)
plt.ylim(-1.5, 1.5)


# Scatter plot
plt.scatter(x, y, alpha = 0.1) # alpha - transparency


# Histogram
plt.hist(x, alpha = 0.3) 
plt.hist(x, bins = 25, alpha = 0.3) # The arguments bins can be used to control the number of intervals


# barplot
plt.bar(cat_var, values, width = 0.35, alpha = 0.3)



# Plot multiple plots
# the first subplot 
plt.subplot(131) # - 1 row, 3 column, as 1st plot             
plt.boxplot(Length)
plt.xticks([])
plt.title('Boxplot of Length')
# the second subplot 
plt.subplot(132) # - 1 row, 3 column, as 2nd plot              
plt.boxplot(Height)
plt.xticks([])
plt.title('Boxplot of Height')
# the third subplot 
plt.subplot(133)
plt.boxplot(Ring) # - 1 row, 3 column, as 3rd plot 
plt.xticks([])
plt.title('Boxplot of Ring')


# Plot multiple in different layout ???
# plots
plt.subplot2grid((3,3), (0,0), colspan=3)
plt.text(0.4, 0.5, 'Row: 0; Col: 0, 1, 2')

plt.subplot2grid((3,3), (1,0), colspan=2)
plt.text(0.4, 0.5, 'Row: 1; Col: 0, 1')

plt.subplot2grid((3,3), (1, 2), rowspan=2)
plt.text(0.2, 0.5, 'Row: 1, 2; Col:2')

plt.subplot2grid((3,3), (2, 0))
plt.text(0.2, 0.5, 'Row: 2; Col: 0')

plt.subplot2grid((3,3), (2, 1))
plt.text(0.2, 0.5, 'Row: 2; Col: 1')



# save img
plt.savefig('pic/matplot.png')





