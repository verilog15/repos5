/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to you under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.logging.log4j.jcl;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.List;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.logging.log4j.core.LoggerContext;
import org.apache.logging.log4j.core.test.appender.ListAppender;
import org.apache.logging.log4j.core.test.junit.LoggerContextSource;
import org.apache.logging.log4j.util.Strings;
import org.junit.jupiter.api.Test;

class LoggerTest {

    @Test
    void testFactory() {
        final LogFactory factory = LogFactory.getFactory();
        assertThat(factory).isInstanceOf(LogFactoryImpl.class);
    }

    @Test
    @LoggerContextSource("LoggerTest.xml")
    void testLog(final LoggerContext context) {
        final Log logger = LogFactory.getLog("LoggerTest");
        logger.debug("Test message");
        verify(context, "o.a.l.l.j.LoggerTest Test message MDC{}" + Strings.LINE_SEPARATOR);
        logger.debug("Exception: ", new NullPointerException("Test"));
        verify(context, "o.a.l.l.j.LoggerTest Exception:  MDC{}" + Strings.LINE_SEPARATOR);
        logger.info("Info Message");
        verify(context, "o.a.l.l.j.LoggerTest Info Message MDC{}" + Strings.LINE_SEPARATOR);
        logger.info("Info Message {}");
        verify(context, "o.a.l.l.j.LoggerTest Info Message {} MDC{}" + Strings.LINE_SEPARATOR);
    }

    private static void verify(final LoggerContext context, final String expected) {
        final ListAppender listApp = context.getConfiguration().getAppender("List");
        final List<String> events = listApp.getMessages();
        assertThat(events).hasSize(1).containsExactly(expected);
        listApp.clear();
    }
}
