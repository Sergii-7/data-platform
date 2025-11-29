FROM postgres:14

# більш надійний apt + встановлення gettext-base
RUN set -eux; \
    # перевести репозиторії на HTTPS (часто вирішує проблеми з портом 80)
    sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list || true; \
    sed -i 's|http://security.debian.org|https://security.debian.org|g' /etc/apt/sources.list || true; \
    apt-get update; \
    apt-get install -y --no-install-recommends gettext-base; \
    rm -rf /var/lib/apt/lists/*

COPY postgresql.conf /etc/postgresql/postgresql.conf
COPY init.sql.template /docker-entrypoint-initdb.d/init.sql.template
COPY init.sh /docker-entrypoint-initdb.d/init.sh

RUN chmod +x /docker-entrypoint-initdb.d/init.sh && \
    chown postgres:postgres /docker-entrypoint-initdb.d/init.sql.template /docker-entrypoint-initdb.d/init.sh

CMD ["postgres", "-c", "config_file=/etc/postgresql/postgresql.conf"]