---
title: "Reproducible journal articles"
author: matt_upson
comments: yes
date: '2015-10-10'
layout: post
modified: 2015-11-11
excerpt: Writing papers for peer review with knitr
published: true
status: publish
tags:
- R
- knitr
- LaTeX
- science
- open data
categories: Rstats
---
 
A few months ago, I wrote a [post](../latex-phd/) about my experiences of writing up a PhD thesis reproducible using $\LaTeX$ and R via [knitr](http://yihui.name/knitr/).
Since then I have started to write up much of my PhD work into academic papers for peer review, so in this post I'm documenting a number of new lessons I have learnt over the last five months.
I'm working on four papers at present, one of which is under review, and I hope to make the source code available as these appear in print.
 
I'm really talking about writing `.Rnw` files, not `.Rmd` files.
Great as I think markdown is, in my mind, it is not quite sophisticated enough (yet perhaps) for writing highly structured documents like journal articles.
These really need $\LaTeX$.
 
In my experience, using these tools for writing papers differs from writing a PhD thesis in two main ways.
Firstly you are likely to produce many more drafts and edits, especially if you are collaborating.
Second, it is likely that you will have more stringent guidelines on how the article can be presented, in line with what the journal or the publisher (and these may not necessarily be the same) require.
 
### Version control
 
<!--[![https://xkcd.com/1296/](http://imgs.xkcd.com/comics/git_commit.png)](https://xkcd.com/1296/)-->
 
 
So I talked about  this before in my previous post, and it is doubly important (to my mind) when you are working on papers.
**Use version control**.
Frankly if you are doing any type of coding, you will save a lot of pain by investing some time in learning how to use some version control software properly.
I use [git](https://git-scm.com/) exclusively, so here I'm really talking about git, but there are other programs available like [subversion](https://subversion.apache.org/).
In what follows, I assume some basic familiarity with these tools, which is easy to pick up from the plethora of articles online, or of course, [SO](http://stackoverflow.com/).
 
Two tips when working with git for academic papers:
 
#### Use branches and tags sensibly
 
I create a new branch every time I receive edits from a co-author (none of whom use $\LaTeX$, but this would be probably more pertinent if they did), and incorporate these edits in this branch.
I may make several commits on this branch, but would squash them into a single commit using `git rebase -i` so that there is just one commit to merge to master.
This keeps your history nice and clean, which is more important if you are planning to make the source code available to others at some point.
 
Tags are something I have only just started to use properly in git, but are extremely useful for writing papers, especially when used in conjunction with [semantic versioning](http://semver.org/).
This is a system for naming software versions to avoid 'software dependency hell', and works on the basis of X.Y.Z (Major.Minor.Patch).
In software usage you would increment Z when you make changes and small edits, Y when making backwardly compatible changes and additions, and X when making significant changes that are no longer backwardly compatible.
 
I've started to adopt this system for writing papers.
The first draft gets named v0.1.0, and I create an annotated tag for this using `git tag -a v0.1.0 -m 'First draft'`.
Minor changes like spelling, grammar, axis labels, etc would merit a Z increment, whilst more comprehensive changes an edits from a co-worker would merit a Y increment.
Once the paper is sent to the journal, I increment the major version, and we get v1.0.0, and comments from the reviewers would merit additional minor versions: v1.1.0, v1.2.0, etc.
 
#### Use gitinfo
 
Now using this kind of system comes into its own when working with the $\LaTeX$ package [gitinfo2](https://www.ctan.org/pkg/gitinfo2).
Gitinfo2 will automatically pick up the version number from tags, and the current branch and commit hash every time you compile.
You can then have it automatically print this at the bottom of every page, or you can pick and choose which information you wish to display.
 
To get started with gitinfo2, you need to install the $\LaTeX$ package, then copy [this file](https://raw.githubusercontent.com/Hightor/gitinfo2/CTAN/post-xxx-sample.txt) into the `.git/hooks/` directory of your repository.
Rename this file to `post-checkout`, or `post-commit` depending on when you want it to run (I use commit).
Once run, this hook will create a file called `.git/gitHeadInfo.gin` from which the package gitinfo2 picks up the relevant details, and uses them in your document.
 
If you're on Linux, you can install this file quickly from your repo using the code below.
Note that you also need to make it executable by giving it the `+x` flag.
 

{% highlight sh %}
wget https://raw.githubusercontent.com/Hightor/gitinfo2/CTAN/post-xxx-sample.txt .git/hooks/post-commit
sudo chmod +x post-commit
{% endhighlight %}
 
Note that to install gitinfo2 on Ubuntu 14.04 I needed to remove the version of teXlive installed from the Ubuntu repository, and do a vanilla install ([full post on this here](http://tex.stackexchange.com/questions/1092/how-to-install-vanilla-texlive-on-debian-or-ubuntu)) 
which allowed me to install new packages with teXlive manager ([tlmgr](https://www.tug.org/texlive/tlmgr.html)).
In windows I could just used the MikTeX package manager, as for some reason gitinfo2 was not installed on the fly when I tried to compile (which is the norm).
 
### Data management
 
I'm a big believer in making things open where possible, and I'm very keen to make my research data open.
I spent a lot of time collecting this data, and I don't want it to disappear because I didn't take some simple steps to prevent this from happening.
 
So I have started to publish my data on [figshare](http://figshare.com/authors/Matthew_Upson/433648).
Figshare is nice, because it will mint you a DOI which offers some protection against your data disappearing into the ether.
The downside is that it is quite hard to properly document your data.
 
A nice alternative is just to use git and publish the data on github.
One particularly appealing feature of this, is that it allows you to track issues with your data, and correct them whilst leaving a log of the changes that you have made through commits, or more formally by issue tracking.
So this is what I have started to do [here](https://github.com/maupson/research_data).
I can create an individual `README.md` for each data file, which will automatically get rendered like [this](https://github.com/maupson/research_data/tree/master/silsoe/silsoe_coarse_root_agg).
 
A couple of additional benefits of using github:
 
* Although I have used figshare for the DOI, you can make your github repository citeable and mint a DOI for it using [zenodo](https://zenodo.org/account/settings/github/). I've done that [here](https://github.com/ivyleavedtoadflax/gapAPI) for an R package I wrote to access data from a REST API.
* If you have spatial data, or site maps, you can render this directly on github by [uploading it in geoJSON format](https://help.github.com/articles/mapping-geojson-files-on-github/).
* You can make use of [shields.io](http://shields.io/) to make funky DOI and license badges like this: [![http://dx.doi.org/10.6084/m9.figshare.1492497](http://img.shields.io/badge/DOI-dx.doi.org%2F10.6084%2Fm9.figshare.1492497-blue.svg)](http://dx.doi.org/10.6084/m9.figshare.1492497)
* The thing I like most of all: once the data is on github, you can call it from your R code directly instead of relying on a local copy. I like this, because it ensures that you are using exactly the same version that you are making available, e.g.:
 

{% highlight r %}
library(dplyr)
library(RCurl)
 
"https://raw.githubusercontent.com/maupson/research_data/master/silsoe/silsoe_soil_organic_carbon/silsoe_soil_organic_carbon.csv" %>%
  getURL %>%
  textConnection %>%
  read.csv %>%
  tbl_df
{% endhighlight %}



{% highlight text %}
## Source: local data frame [216 x 15]
## 
##    year   ID block ctrltmt treat dist_m depth_cm di_cm OCC_g_100g BD_g_cm3
## 1  2011 1CB4    B1     tmt     C    0.5        5    10  5.5246218     1.17
## 2  2011 1CB4    B1     tmt     C    0.5       16    10  2.7800000     1.44
## 3  2011 1CB4    B1     tmt     C    0.5       30    20  2.4900000     1.48
## 4  2011 1CB4    B1     tmt     C    0.5       50    20  1.1465367     1.51
## 5  2011 1CB4    B1     tmt     C    0.5       83    45  0.4992435     1.54
## 6  2011 1CB4    B1     tmt     C    0.5      128    45  0.2996962     1.56
## 7  2011 1CB4    B1     tmt     C    1.5        5    10  3.3829914     1.26
## 8  2011 1CB4    B1     tmt     C    1.5       16    10  3.0000000     1.28
## 9  2011 1CB4    B1     tmt     C    1.5       30    20  2.4900000     1.47
## 10 2011 1CB4    B1     tmt     C    1.5       50    20  0.4979960     1.79
## ..  ...  ...   ...     ...   ...    ...      ...   ...        ...      ...
## Variables not shown: mEquiv (int), mMeas (dbl), tC (dbl), SOC_ESM_Mg_ha2
##   (dbl), SOC_Mg_ha2 (dbl)
{% endhighlight %}
 
### Endfloat
 
Something I discovered recently which can save a little bit of time, is the [endfloat](https://www.ctan.org/pkg/endfloat) package.
This $\LaTeX$ package simplifies the onerous task of moving figures and tables to the end of a manuscript, and will create a list of table and figure captions.
Some journals no longer require this, but many still do.
 
### Be kind to co-authors
 
All of the co-authors I have written with to date have not been proficient in $\LaTeX$, which means that they will make comments either on paper copies or on pdfs. 
This can be annoying for them, as they lose access to their comments after they hand over paper copies.
Just to remind them of the substantive comments that they have made on the previous version, I use the [todonotes](https://www.ctan.org/pkg/todonotes) package to both create inline comments within the text, and a list of todos at the beginning of the document.
 
It is possible to compare two `.tex` files, and produce a comprehensive list of changes using [latexdiff](https://www.ctan.org/pkg/latexdiff), which can be integrated with git using a wrapper: [git-latexdiff](https://gitlab.com/git-latexdiff/git-latexdiff).
This is on my list of things to set up, but I've not yet had a need to.
 
### Submitting
 
So fast-forward to that magical moment when you are ready to submit.
Elsevier and Springer both use editorial manager, which has so far been much easier to use than expected (or documented online).
The only major changes I have made to my workflow to accommodate it, is to ensure that figures get produced in the root of the project, rather than in a `figure/` folder.
This is necessary if submitting by zip file, but not clearly if you submit files manually, one by one.
In addition, you may also need to ensure that knitr appends the file extension to the end of the graphics files, which is not the default behaviour (and is not usually required by $\LaTeX$).
This can be achieved using the code below.
 

{% highlight r %}
library(knitr)
 
opts_chunk$set(
  fig.path = "",
  dev = "pdf"
  )
 
# Allow knitr to print graphics suffixes, from
# https://github.com/yihui/knitr-examples/blob/master/033-file-extension.Rnw
 
hook_plot = knit_hooks$get('plot')
knit_hooks$set(
  plot = function(x, options) {
   # if x is foo.pdf, make it foo.pdf.whatever so the plot hook removes the extension internally
  x = paste(c(x, 'whatever'), collapse = '.')
  hook_plot(x, options)
})
{% endhighlight %}
 
### To sum up
 
So, a few more nuggets of information that may be useful.
One thing I would say is that at the end of the day, no matter how smart you are with your manuscripts, this does not protect you from inconsistencies on the part of the journal.
 
I was asked recently to provide a manuscript in double line spacing, even though I had used the publisher's template and enabled double spacing within it in the way that was recommended.
In the end, I used a really hacky fix to make the whole thing look awful, but with sufficiently wide spacing to be accepted for peer review.
And this is not the worst case I have heard, but I hope that as the idea of open and reproducible research becomes more widespread, that journals will become better informed about the requirements for documents set out by the publisher.
 
