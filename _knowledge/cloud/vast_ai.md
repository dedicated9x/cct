# SSH

3. Utworzenie kluczy (na lokalnej maszynie)
ssh-keygen -t rsa -b 4096 -f ~/.ssh/my_new_vast_key
2. Wkleic .pub do interfejsu z vast.ai
3. Połączyć się :)
ssh -i ~/.ssh/my_new_vast_key -p 44614 root@75.157.149.187 -L 8080:localhost:8080

# Github