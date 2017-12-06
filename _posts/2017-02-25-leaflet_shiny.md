---
title: "Shiny interactive map app development"
author: matt_gregory
comments: yes
date: '2017-02-25'
modified: 2017-03-20
layout: post
excerpt: "The dangers of cherry picking evidence"
published: TRUE
status: processed
tags:
- leaflet
- shiny
- mapping
- maps
categories: Rstats
output: html_document
---
 

 
I've been [developing](http://www.machinegurning.com/rstats/uk_obesity/) my mapping skills in R for a couple of months now. I started off using a couple of different packages but was quickly drawn to `leaflet` which allows one to create interactive web maps with the [JavaScript Leaflet Library](http://leafletjs.com/). It has loads of great features out of the box and will meet most of your mapping needs (if old fashioned people reject your interactive map you can also provide a [static version](http://stackoverflow.com/questions/31336898/how-to-save-leaflet-in-rstudio-map-as-png-or-jpg-file) using the `mapshot` function in the `mapview` package available on CRAN). 
 
This blog post aims to persuade you as to the merits of `leaflet` especially when combined with [shiny](https://shiny.rstudio.com/) facilitating the development of some powerful apps for your users to explore data with. To demonstrate I simulate some data and pose a question: how do `X1` and `X2` vary between schools within Local Education Authorities (LA) in England?  
 
# App - Cherry Picker
 
A Shiny app for mapping School specific variables by Local Education Authority with Leaflet in R.
 
## The live app
 
[App-cherry_picker](https://mammykins.shinyapps.io/App-cherry_picker/) allows the user to investigate correlations between two made-up variables (we call them `apples` and `pears` [stairs], more interesting names than `X1` and `X2`) drawn from a uniform distribution. 
It demonstrates how [cherry picking](https://en.wikipedia.org/wiki/Cherry_picking), or finding data that supports your viewpoint (and ignoring data that does not) can be easily done even with two totally uncorrelated variables. We also use this appletunity for fruit related puns or [punnet](https://en.wikipedia.org/wiki/Punnet)-try.  
 
## Use the app
 
Give it a try: [App-cherry_picker](https://mammykins.shinyapps.io/App-cherry_picker/).  
 
## Clone the [code](https://github.com/mammykins/App-cherry_picker)
 
The code looks complicated right? Shiny can be tricky as it can be more difficult to debug than regular R. However, it's not so bad, the code is developed iteratively, it's not written in one session. Make sure you know what the end [user wants](https://en.wikipedia.org/wiki/Usability_testing) and show it to them frequently for feedback. Gradually your app 'll evolve to meet the needs of the user. The interactivity is worth the extra effort and there are plenty of [good tutorials](https://shiny.rstudio.com/tutorial/) available!  
 
I started this app development by going to the Shiny gallery and copying code from an [app](https://shiny.rstudio.com/tutorial/) which looked similar to the desired app I had in mind. You can do the same by copying my code as a starting point.    
 
## Use version control
 
Prior to app development of your own, make sure you are using version control such as git. Use Github (a repository hosting site; see [here for the difference between git and Github](http://stackoverflow.com/questions/13321556/difference-between-git-and-github)) or other sites to share your code. This will make life a lot easier as explained [here](https://swcarpentry.github.io/git-novice/).
 
## Shiny is special
 
Shiny is a package that contains a set of functions to build web applications within R. However, bear in mind:  
 
* for the correct use of its functions, files need to be organised appropriately (note the ui, server and global files found within an [App prefixed folder](https://github.com/mammykins/App-cherry_picker)).  
* the syntax of its functions are slightly different to base R.  
 
Briefly the `ui.R` is the app's frontend and the `server.R` is the backend ([more detail](https://shiny.rstudio.com/tutorial/lesson1/)). `global.R` has its own niche uses such as pre-loading packages and data that can be used both in `ui.R` and `server.R`. For example, in my app I use it to read in all the core datasets and prepare the mapping data. I also call a [custom function](https://github.com/mammykins/App-cherry_picker/blob/master/make_popup_vector_from_numeric.R) which is used to plot a school Marker on the Leaflet map.  
 
### UI
 
Here we have one of the tabs that you can navigate to. This `tabPanel` is one of four in the app that are contained within the `navbarPage` (see [ui.R](https://github.com/mammykins/App-cherry_picker/blob/master/ui.R)). Each `tabPanel` is seperated from one another by commas. It's important to lay out your code sensibly to avoid misplacing a comma, fortunately R Studio takes care of this for you.    
 
Note the use of HTML builder functions (e.g. `p()` and `h4()`) for constructing HTML documents (the output is an intereactive webpage).
 

{% highlight r %}
tabPanel("Fruit limit",
         h4("Schools selected from Data explorer tab"),
         DT::dataTableOutput("green_grocers"),
         plotOutput("scatter_fruit", height = 200),
         h4("How many items of fruit required to feed the Schools of interest?"),
         p("This app was developed to showcase a Shiny app in R using the Leaflet package. Use your imagination, make a map!",
           style = "font-family: 'times'; font-si16pt")
         )
{% endhighlight %}
 
The `tags` environment also contains convenience functions. Here I insert the [Machine Gurning logo](https://github.com/mammykins/App-cherry_picker/blob/master/www/mg_logo.png) (which must be saved in a folder called [www](https://github.com/mammykins/App-cherry_picker/tree/master/www)) and produce a quote within the user interface.  
 

{% highlight r %}
tags$a(img(src = "mg_logo.png", height = 144, width = 144),
 
       href = "http://www.machinegurning.com/posts/"),
 
tags$blockquote("Correlation does not imply causation.", cite = "Anon.")
{% endhighlight %}
 
### Reactivity
 
Shiny is all about reactivity. The user picks a new LA of interest and the app adapts. The `selectInput` is the important feature here that provides teh user the choice of LA and phase of schools. The user is provided with a list of schools through the `global.R` assigned object `la_user_friendly_list`. They pick the LA and that string can now be called by the `server.R` using `input$la_of_interest`.
 

{% highlight r %}
                                      selectInput(inputId = "la_of_interest",
                                                  label = "Local Authority",
                                                  choices = la_user_friendly_list,
                                                  selected = "202 - Camden"),
                                      selectInput(inputId = "phase", label = "School phase",
                                                  choices = c("Secondary", "Primary"))
{% endhighlight %}
 
We use this information to filter our fruits (`apples` and `pears` joined data from `global.R`) all-schools dataframe (including geospatial goordinates) and create a reactive object called `ks4_to_map()` (this is vestigial name). In the following code we filter our dataframe by splitting the input string and extract the numeric LA Code (e.g. "202 - Camden" to 202). Thus any schools of that LA are filtered and further filtered by school phase and complete cases.  
 

{% highlight r %}
#  NOTE input$la_of_interest, not just la_of_interest; this is from the ui.R inputID
  ks4_to_map <- reactive({
    dplyr::filter(fruits,
                  la_number == as.numeric(sapply(strsplit(input$la_of_interest[[1]], split = " "), "[[", 1))) %>%
      left_join(school_locations,  #  we need geospatial data
                by = c("urn", "la_number")) %>%
      dplyr::filter(phase == input$phase) %>%
      mutate(easting = easting.x, northing = northing.x) %>%
      select(-easting.y, -northing.y) %>%
      na.omit()
  })
{% endhighlight %}
 
Great, we have the schools we need to plot in our LA. To provide the national perspective we can draw a histogram of all the school data using our `global.R` assigned `fruits` dataframe object and then use our reactive object `ks4_to_map` to add a `rug()` of blue ticks to the bottom. Note how we assign this object, which is based on a reactive object, to `output$hist_apples`. `renderPlot` renders a reactive plot that is suitable for assigning to an output slot.
 

{% highlight r %}
  # Provide variable distributions to aid comparison to rest of the country
  output$hist_apples <- renderPlot({
    hist(fruits$apples,  #  Note this does all schools, doesn't filter for School Phase, which you may want
         main = "Apple of my eye",  #  Note how we call our pre-filtered data assigned in global!
         xlab = "Apples", col = "salmon", border = 'white', xlim = c(0, 1))
    rug(ks4_to_map()$apples, ticksize = -0.2, lwd = 3, col = "blue")  #  For the rug we use our reactive dataframes
  })
{% endhighlight %}
 
Thus to render this object in the `ui.R` we use the shiny function `plotOutput`. You can find this line of code in the `absolutePanel` in the `ui.R`.  
 

{% highlight r %}
plotOutput("hist_apples", height = 200)  #  Note the use of "output_name"
{% endhighlight %}
 
### Leaflet
 
The same principles apply for creating the map, the `ks4_to_map()` reactive object is transformed into a spatial points dataframe object called `ks4_sp_ll`. Importantly we need to convert the esoteric northings and eastings, used by OS in the UK, to latitude and longitude. The credit for this solution should go to this [tutorial](http://www.alex-singleton.com/R-Tutorial-Materials/7-converting-coordinates.pdf).  
 

{% highlight r %}
   # LEAFLET -----------------------------------------------------------------
  
  # Create coordinates variable, first argument
  # Create the SpatialPointsDataFrame, note coords and data are distinct slots in S4 object
  # Vestigial name from condainment app
  ks4_sp_ll <- reactive({
    #  we use x as a placeholder just within this reactive bit, helps with the last renaming step
    x <- spTransform(
      sp::SpatialPointsDataFrame(dplyr::select(ks4_to_map(), easting, northing),
                                 data = dplyr::select(ks4_to_map(), -easting, -northing),
                                 proj4string = CRS("+init=epsg:27700")),
      CRS(latlong)
    )
    # Convert from Eastings and Northings to Latitude and Longitude and rename columns
    colnames(x@coords)[colnames(x@coords) == "easting"] <- "longitude"
    colnames(x@coords)[colnames(x@coords) == "northing"] <- "latitude"
    
    x
  })
{% endhighlight %}
 
Now we render the reactive leaflet object by assigning the output slot `output$mymap` using the function `renderLeaflet`. The [online help](https://rstudio.github.io/leaflet/) is very good for Leaflet so I won't go into all the details here. In the interactive map you can change the map tile set, plot circles for the different variables on and off, have the apples displayed and the pears variable as an outline, use markers to identify school names on the maps.  
 

{% highlight r %}
    # LEAFLET -----------------------------------------------------------------
  
  output$mymap <- renderLeaflet({
    #  See the full code in server.R
  })
{% endhighlight %}
 
## DT
 
The R package [DT](https://rstudio.github.io/DT/) provides an R interface to the JavaScript library DataTables. R data objects (matrices or data frames) can be displayed as tables on HTML pages, and DataTables provides filtering, pagination, sorting, and many other features in the tables. I make use of this in the second tab called "Data Explorer". You can create all your standard Excel-like dashboard features that people like. Including currency symbols, colouring fonts (blue for good, red for bad) etc.  
 
This package also facilitates row selection, so the user can click on schools of interest and explore them further. Here we use selected rows to help us in our cherry picking exercise. By clicking on potential cherries (i.e. where both apples and pears are relatively high) in the "Data Explorer" tab, this then draws red circles on a scatterplot of all schools' apples and pears data for those highlighted schools. Does your selected school make it into the cherry picking region?  
 
As you interact with the DT table (e.g.sort columns, search the table, or navigate through pages), DT will expose some information about the current state of the table to Shiny. At the moment, this information is available in the input object of the Shiny server function (suppose the table output id is `tableId`). The important bit in the code is the suffixing of `_rows_selected` on the `tableId` (in this case creating `fruit_table_data_rows_selected`), [explained here](https://rstudio.github.io/DT/shiny.html).  
 

{% highlight r %}
# ANOTHER TAB ----------------------------------------------------------
  # Here we can use another tab to display some furter analysis or statistics
# https://rstudio.github.io/DT/shiny.html
  # row selection
  output$green_grocers <- DT::renderDataTable(datatable(
    slice(ks4_to_map() %>%
            select(school_name, apples, pears, urn) %>%
            arrange(desc(apples)) %>%
            rename(School_Name = school_name, URN = urn), #  creates identical table to slice from, see fruit_table_data
          input$fruit_table_data_rows_selected) %>%  
      mutate(made_up_statistic = (apples + pears) * (if_else(input$phase == "Secondary",
                                              3,  #  Secondary school children need more fruit!?
                                              1) * 30),
             cherry_status = round((apples + pears) / 2, 2)
    ) %>%  #  we refine the datatable here and prettify
      select(School_Name, cherry_status, made_up_statistic)
  ) %>%  #  and prettify
    formatRound(c("School_Name", "made_up_statistic"),
                0)
  )
{% endhighlight %}
 
 
### Other features
 
People like Excel, let them save app outputs so that they can then open them in Excel.  
 

{% highlight r %}
  output$download_data <- downloadHandler(
    filename = function() { paste("cherry_picker_app_", input$la_of_interest, '.csv', sep = '') },
    content = function(file) {
      write.csv(ks4_to_map(), file)
    }
  )
{% endhighlight %}
 
 
 
### Try it yourself
 
Try [cloning my repo](https://help.github.com/articles/cloning-a-repository/) and replacing the data within the apples and pears columns in both of the respective csv files with some school data of interest (match on the URN). Then run the [school_data_write_as_rds.R](https://github.com/mammykins/App-cherry_picker/blob/master/school_data_write_as_rds.R) to write the data as the more memory efficient RDS (there's no reason your app can't directly read in the data as a csv). Now you have a similar app that will produce a map of your two variables of interest; albeit still called apples and pears!  From this starting point you can continue to add and remove features to get your app into the user desired shape.  
 
## Conclusion
 
> The future's bright, the future's Shiny!
 
Shiny map apps are awesome and can be useful for spotting patterns in data that might be worth a more robust statistical follow up investigation. It's important to protect oneself from cherry picking for a unconciously favoured hypothesis as demonstrated with this app, where one could pick out numerous schools that are high for both apples and pears. If one only presented these schools in a discussion, one might think that there was good evidence for banning apples in schools. At the least to prevent everything from going pear shaped!  
 

{% highlight r %}
devtools::session_info()
{% endhighlight %}



{% highlight text %}
##  setting  value                       
##  version  R version 3.3.2 (2016-10-31)
##  system   x86_64, mingw32             
##  ui       RStudio (1.0.136)           
##  language (EN)                        
##  collate  English_United Kingdom.1252 
##  tz       Europe/London               
##  date     2017-03-20                  
## 
##  package    * version date       source                                   
##  checkpoint   0.3.18  2016-10-31 CRAN (R 3.3.2)                           
##  devtools     1.12.0  2016-06-24 CRAN (R 3.3.2)                           
##  digest       0.6.11  2017-01-03 CRAN (R 3.3.2)                           
##  evaluate     0.10    2016-10-11 CRAN (R 3.3.1)                           
##  knitr        1.15.1  2016-11-22 CRAN (R 3.3.2)                           
##  magrittr     1.5     2014-11-22 CRAN (R 3.3.1)                           
##  memoise      1.0.0   2016-01-29 CRAN (R 3.3.1)                           
##  rmd2md       0.1.4   2017-01-16 Github (ivyleavedtoadflax/rmd2md@e2d6ae4)
##  stringi      1.1.2   2016-10-01 CRAN (R 3.3.1)                           
##  stringr      1.1.0   2016-08-19 CRAN (R 3.3.1)                           
##  withr        1.0.2   2016-06-20 CRAN (R 3.3.1)
{% endhighlight %}
