"""Catálogo del corpus: qué documentos forman la base de conocimiento.

Se eligen solo fuentes oficiales y públicas sobre los temas del Avance 1:
errores de arranque, configuración (properties/yml), dependencias Maven
y logs. Así cada respuesta se puede citar con una URL verificable.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    id: str        # nombre corto, también es el nombre del archivo local
    title: str     # título legible, se usa para citar la fuente
    url: str
    doc_type: str  # "html" o "pdf"

    @property
    def filename(self) -> str:
        return f"{self.id}.{self.doc_type}"


SB = "https://docs.spring.io/spring-boot"
SF = "https://docs.spring.io/spring-framework/reference"
MVN = "https://maven.apache.org/guides/introduction"

SOURCES = [
    # --- Spring Boot: configuración y arranque ---
    Source("sb_external_config", "Spring Boot - Externalized Configuration", f"{SB}/reference/features/external-config.html", "html"),
    Source("sb_profiles", "Spring Boot - Profiles", f"{SB}/reference/features/profiles.html", "html"),
    Source("sb_logging", "Spring Boot - Logging", f"{SB}/reference/features/logging.html", "html"),
    Source("sb_auto_config", "Spring Boot - Auto-configuration", f"{SB}/reference/using/auto-configuration.html", "html"),
    Source("sb_web_servlet", "Spring Boot - Servlet Web Applications", f"{SB}/reference/web/servlet.html", "html"),
    Source("sb_actuator_endpoints", "Spring Boot - Actuator Endpoints", f"{SB}/reference/actuator/endpoints.html", "html"),
    Source("sb_sql", "Spring Boot - SQL Databases", f"{SB}/reference/data/sql.html", "html"),
    # --- Spring Boot: guías how-to (problemas típicos) ---
    Source("sb_howto_webserver", "Spring Boot How-to - Embedded Web Servers", f"{SB}/how-to/webserver.html", "html"),
    Source("sb_howto_properties", "Spring Boot How-to - Properties and Configuration", f"{SB}/how-to/properties-and-configuration.html", "html"),
    Source("sb_howto_data_access", "Spring Boot How-to - Data Access", f"{SB}/how-to/data-access.html", "html"),
    Source("sb_maven_plugin", "Spring Boot Maven Plugin", f"{SB}/maven-plugin/index.html", "html"),
    # --- Spring Framework ---
    Source("sf_dependencies", "Spring Framework - Dependency Injection", f"{SF}/core/beans/dependencies/factory-collaborators.html", "html"),
    Source("sf_transactional", "Spring Framework - Using @Transactional", f"{SF}/data-access/transaction/declarative/annotations.html", "html"),
    # --- Maven ---
    Source("mvn_dependency", "Maven - Dependency Mechanism", f"{MVN}/introduction-to-dependency-mechanism.html", "html"),
    Source("mvn_lifecycle", "Maven - Build Lifecycle", f"{MVN}/introduction-to-the-lifecycle.html", "html"),
    Source("mvn_pom", "Maven - Introduction to the POM", f"{MVN}/introduction-to-the-pom.html", "html"),
    # --- Java (PDF) ---
    Source("java_troubleshooting", "Java SE 21 - Troubleshooting Guide", "https://docs.oracle.com/en/java/javase/21/troubleshoot/troubleshooting-guide.pdf", "pdf"),
]
