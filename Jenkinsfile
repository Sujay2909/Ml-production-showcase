// Jenkinsfile — Declarative Pipeline
// Mirrors the CI/CD setup that reduced release cycles from weeks to 48 hours.

pipeline {
    agent {
        docker {
            image 'python:3.11-slim'
            args '-v /var/run/docker.sock:/var/run/docker.sock'
        }
    }

    environment {
        APP_ENV        = 'test'
        LOG_LEVEL      = 'WARNING'
        MLFLOW_URI     = credentials('mlflow-tracking-uri')
        SLACK_WEBHOOK  = credentials('slack-webhook-url')
    }

    options {
        timeout(time: 45, unit: 'MINUTES')
        buildDiscarder(logRotator(numToKeepStr: '10'))
        timestamps()
    }

    stages {
        stage('Setup') {
            steps {
                sh '''
                    pip install --upgrade pip
                    pip install -r requirements-dev.txt
                    python -m spacy download en_core_web_sm
                '''
            }
        }

        stage('Lint') {
            parallel {
                stage('Black') {
                    steps { sh 'black --check src/ tests/' }
                }
                stage('isort') {
                    steps { sh 'isort --check-only src/ tests/' }
                }
                stage('Flake8') {
                    steps { sh 'flake8 src/ tests/ --max-line-length=100' }
                }
                stage('Mypy') {
                    steps { sh 'mypy src/ --ignore-missing-imports' }
                }
            }
        }

        stage('Test') {
            steps {
                sh '''
                    pytest tests/ \
                        --cov=src \
                        --cov-report=xml \
                        --cov-report=html \
                        --cov-fail-under=70 \
                        --junitxml=test-results.xml \
                        -v
                '''
            }
            post {
                always {
                    junit 'test-results.xml'
                    publishHTML(target: [
                        reportDir  : 'htmlcov',
                        reportFiles: 'index.html',
                        reportName : 'Coverage Report'
                    ])
                }
            }
        }

        stage('Build Docker') {
            when { branch 'main' }
            steps {
                sh """
                    docker build \
                        --target runtime \
                        --tag ml-platform:${env.BUILD_NUMBER} \
                        --tag ml-platform:latest \
                        .
                """
            }
        }

        stage('Model Validation') {
            when { branch 'main' }
            environment {
                MLFLOW_TRACKING_URI = "${env.MLFLOW_URI}"
            }
            steps {
                sh 'pytest tests/test_ml_pipeline.py -v --tb=short'
            }
        }

        stage('Deploy') {
            when { branch 'main' }
            steps {
                sh '''
                    docker-compose up -d --no-build api
                    sleep 15
                    curl -f http://localhost:8000/health || exit 1
                '''
                echo "Deployment complete — release cycle < 48 hours achieved"
            }
        }
    }

    post {
        success {
            slackSend(
                channel: '#ml-platform',
                color: 'good',
                message: "✅ *Build #${env.BUILD_NUMBER}* passed on `${env.BRANCH_NAME}`"
            )
        }
        failure {
            slackSend(
                channel: '#ml-platform',
                color: 'danger',
                message: "❌ *Build #${env.BUILD_NUMBER}* failed on `${env.BRANCH_NAME}` — ${env.BUILD_URL}"
            )
        }
        cleanup {
            cleanWs()
        }
    }
}
