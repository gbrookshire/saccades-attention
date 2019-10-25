# Crop images
convert raw/0.jpg -crop 300x300+250+40 cropped/0.jpg
convert raw/1.jpg -crop 420x420+100+2  cropped/1.jpg
convert raw/2.jpg -crop 360x360+130+5  cropped/2.jpg
convert raw/3.jpg -crop 440x440+20+20  cropped/3.jpg
convert raw/4.jpg -crop 360x360+120+80 cropped/4.jpg
convert raw/5.jpg -crop 460x460+100+5 cropped/5.jpg

# Resize to a common size
for filename in cropped/*.jpg; do
    convert $filename -resize 256x256! $filename
done

# Print info to check that it worked
identify cropped/*.jpg
