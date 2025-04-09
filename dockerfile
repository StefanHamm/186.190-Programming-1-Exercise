
# Use a base image with linux
FROM ubuntu:latest

# install pdflatex
RUN apt-get update && \
    apt-get install -y texlive-latex-base texlive-fonts-recommended texlive-fonts-extra texlive-latex-extra

#install perl 
RUN apt-get update && apt-get install -y --no-install-recommends \
    perl \
    build-essential \
    cpanminus \
    && rm -rf /var/lib/apt/lists/* \
    && cpanm Text::CSV \
    && rm -rf /root/.cpanm # Clean up cpanm cache (optional but good practice)
